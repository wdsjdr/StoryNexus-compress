"""启发式事实提取（M12：题材 profile 化 + 线索物件层 + 卡先验）。

替代旧版无约束正则（`([汉字]{2,4})的(...)` / `([汉字]{2,4})(动词)`）——
旧版会把任意文本片段（"远比表面""上去要""潮水般涌"）当作主语，产生垃圾
三元组污染 CSA 上下文。

M12 架构：
  1. FactProfile 双档：cultivation（修仙/权谋事件流，现状谓词表）/
     slice（日常文：对话约定 + 信物/线索物件 + 关系推进）。
     SkillSpec.fact_profile 字段驱动（YAML 可配）。
  2. 实体发现四证据：
     称谓后缀 / 姓氏人名 / 高频跨章人名（无姓氏，第 3 证据）/ 卡名先验
     另有 motif 线索物件层（高频 + 跨章 + 线索尾缀，仅作宾语）。
  3. 事实提取：主语必须 ∈ 实体集；宾语 ∈ 实体∪motif；谓词白名单；
     优先级 关系/对话 > 动作 > 拥有；每章去重且限条数；**不再提取"登场"**。

供三处使用：M7 导入管线（application.confirm_import）、写章后启发式兜底
（orchestrator）、1M 压测脚本（scripts/benchmark_1m.py）。
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from app.domain.models.timeline import FactTriple

# ── 称谓后缀：主体(2-4字)+称谓 = 高置信实体 ──
TITLE_SUFFIX = (
    "仙君|仙子|大帝|真人|老祖|娘娘|大师|长老|宗主|圣女|公子|夫人|姐姐|"
    "师妹|师弟|师兄|师叔|前辈|尊者|道君|府主|城主|宫主|圣主|家主|掌门|"
    "国师|将军|太子|皇子|公主|王妃|皇后|皇上|姑娘|小姐"
)
_TITLE_RE = re.compile(
    rf"(?<![\u4e00-\u9fff])([\u4e00-\u9fff]{{2,4}}(?:{TITLE_SUFFIX}))"
)

# ── 重叠窗口频率 / 整词边界验证（截断片段过滤） ──
_OVERLAP_2 = re.compile(r"(?=([\u4e00-\u9fff]{2}))")
_OVERLAP_3 = re.compile(r"(?=([\u4e00-\u9fff]{3}))")
_WORD_BOUNDED = re.compile(
    r"(?<![\u4e00-\u9fff])([\u4e00-\u9fff]{2,3})(?![\u4e00-\u9fff])"
)
# 左边界出现（词首前非汉字）：motif 碎片判定——
# 璃罐/荷糖/火虫 永远嵌在 玻璃罐/薄荷糖/萤火虫 内部（词首前必为汉字），
# 而真实信物（红绳/薄荷糖）常在标点/句首后独立出现。
_LEFT_BOUNDED_RE = re.compile(r"(?<![\u4e00-\u9fff])([\u4e00-\u9fff]{2,3})")

# ── 停用字符 / 尾部动词 / 高频非专名排除表（保持 M10 精确优先约束） ──
_STOP_CHARS = frozenset(
    "的了着在是这那和与就也很不也没我你他她它们将把被从向对为以于之而或及"
    "等个种其所有些因如若却已正更最刚又再仍只但且或除并既一来去出上中下大小"
    "多少最常般样话处点面分间时个各每哪位怎什么几两三地"
)
_TAIL_VERBS = frozenset(
    "看说问道想知感受轻声沉冷笑点头摇头站望盯听退躲藏救杀走来往去握问答应疑"
    "皱起抬头低头转身伸手开口沉默低语呢喃叹息点头微笑皱眉进出回到入上下离"
    "跑跳飞追赶送交给拿取找寻见遇打击败斩灭护帮扶拉拖推抱背带领引跨迈踏踩跃"
    "冲闯逃停留守等化变转改移看视探察观啊呢吗吧哟哦呀咦嘿呵么叫喊喝闻"
    "曰怒喜惊惧慌忧悲"
)
# 组合名扩展的首/尾字过滤（"见八戒/救唐僧"动词头、"唐僧乃/行者闻"虚词/动词尾）
_EXPAND_BAD_HEAD = frozenset(
    "见望请说救教唤叫喊问带劝报献送迎寻讨拿把将领率使派遣差请随蒙奉"
    "好老小大假保这那有各几每此彼其某闻投今即遂乃亦必自皆又但且令"
)
_EXPAND_BAD_TAIL = frozenset(
    "乃闻见使依即心忍急慌才接施暗近随骂喝挑牵指教望奉蒙传问答曰道叫喊皇帝"
    "呢吗吧么哟呀咦嘿呵坐站住睡哭笑喊一二三四五六七八九十百千万"
    "亦令先唤必自谓辞遂皆今即兵奸城人请辞"
) | _TAIL_VERBS | _STOP_CHARS
# 强证据 3 绕行的首停用字白名单（停用表中可作人名头的古典名：如/哪/太）
_STRONG3_HEAD_CHARS = frozenset("如哪太")
_EXCLUDE_WORDS = frozenset(
    "一声|起来|出来|过来|下去|问道|笑道|冷声|轻声|柔声|沉声|淡声|修士|气息|"
    "力量|东西|事情|一般|时候|主人|颤抖|一下|无比|身上|少顷|此刻|下来|"
    "感受|心中|嘴角|脸上|眼中|声音|男人|女人|话语|心神|随即|当下|气血|神魂|"
    "神色|目光|眼神|表情|语气|身体|手掌|手臂|肩膀|大腿|腰肢|双腿|身影|面容|"
    "动作|伤势|指尖|体内|手中|胸前|眼底|衣袖|长袍|衣物|空间|天地|世界|时间|"
    "地方|瞬间|样子|感觉|想法|意思|机会|办法|方法|消息|情况|原因|结果|问题|"
    "方向|范围|程度|态度|心情|眼泪|脸色|嘴唇|灵气|法力|功法|丹药|灵石|境界|"
    "修为|神识|阵法|法宝|剑意|拳意|威压|气势|杀气|剑光|刀光|光芒|火焰|寒气|"
    "煞气|魔气|仙气|妖气|血气|死气|生机|气机|气运|因果|天道|规则|法则|本源|"
    "虚空|领域|禁制|傀儡|妖兽|妖王|凡人|武者|剑修|丹修|体修|魔修|邪修|散修|"
    "弟子|姑娘|夫人|大人|大王|阁下|朋友|兄弟|姐妹|父母|儿女|夫君|妻子|小妾|"
    "丫鬟|仆从|手下|随从|护卫|侍卫|对手|敌人|仇人|伙伴|同伴|同门|同辈|长辈|"
    "晚辈|少年|青年|中年|老年|小孩|老人|大汉|胖子|少女|女子|妇人|老妇|丫头|"
    "娃娃|孩童|声音|身影|掌风|拳风|剑风|剑气|灵力|法力|真气|真元|元力|"
    "血气|精血|元神|元婴|金丹|筑基|练气|化神|炼虚|合体|大乘|渡劫|飞升|成仙|"
    "灵根|道体|圣体|血脉|传承|记忆|意识|意志|神魂|灵魂|肉身|筋骨|脏腑|血液|"
    "伤口|伤势|剧痛|眩晕|昏迷|清醒|苏醒|沉默|寂静|安静|喧嚣|吵闹|"
    "突然|果然|随后|然后|毕竟|可惜|闻言|说罢|只见|顿时|立刻|马上|终于|"
    "竟然|居然|仿佛|似乎|恐怕|大概|也许|看来|想来|想必|其实|原来|本来|"
    "终究|到底|难道|反而|倒是|反正|当然|的确|不过|可是|只是|还是|就是|"
    "既然|所以|因为|因此|于是|接着|跟着|旁边|身后|眼前|内心|脑海|脑子|"
    "心里|表面|面上|嘴上|背后|面前|周围|附近|一边|另一边|这边|那边|"
    "哈哈|呵呵|嘿嘿|嘻嘻|噗嗤|哎呀|哎哟|嗯嗯|咳咳|罢了|也罢|好了|算了|"
    "得了|糟糕|完了|坏了|对了|此刻|此时|这般|如此|这样|那样|怎么|什么|"
    "为何|如何|多少|多么|其它|其余|其中|其实|一样|一种|一起|一时|一眼|"
    "一手|一脸|满眼|满口|满心|全然|完全|根本|绝对|实在|特别|十分|非常|"
    "极其|极为|格外|愈发|越来|越来越|一瞬|片刻|良久|许久|半晌|须臾|"
    "哈哈哈|嗯啊|啧啧|哦哦|说完|否则|到时候|无妨|好徒儿|道侣|仙子|"
    "好好好|啊啊啊|真的吗|片刻后|放心吧|狠狠地|要不然|轰隆隆|好不好|"
    "谢谢|再见|晚安|早安|抱歉|对不起|没关系|别客气|不客气|"
    "轻柔地|怎么回事|怎么办|没什么|没想到|不可能|是不是|行不行|干什么|"
    "凭什么|怎么样|为什么|哪知道|谁知道|不知道|知不知道|"
    "一下子|一会儿|突然就|慢慢地|缓缓地|轻轻地|静静地|默默地|"
    "娘子|师兄|师姐|师妹|师弟|师尊|师父|师傅|老祖|娘娘|教主|长老|公子|"
    "小姐|夫人|姑娘|大哥|小弟|前辈|道友|道兄|大姐|大叔|大爷|阿姨|叔叔|"
    "伯伯|祖母|祖父|奶奶|爷爷|哥哥|姐姐|弟弟|妹妹|老婆|老公|丈夫|小人|"
    "老夫|老身|贫道|贫僧|小子|丫头|娃儿|女修|男修|门人|下人|上人|"
    "何事|何况|另外|古怪|尸体|房门|门外|施主|明日|温热|舒服|舒服吗|莫非|"
    "调教|都不行|练功|双修|传功|后方|外界|城外|后宫|盛世|山神|"
    "公主|国师|王爷|侯爷|宗主|宫主|家主|护法|高人|圣子|太子|皇上|皇后|"
    "阁主|堂主|洞主|城主|府主|掌门|方丈|族长|判官|龙王|真君|道君|帝君|"
    "魔子|妖帝|妖王|妾身|奴家|女施主|状元|知府|县令|将军|元帅|先锋|太医|"
    "丫鬟|侍女|宫女|太监|侍卫|士兵|捕快|媒婆|农夫|渔夫|车夫|更夫|铁匠|"
    "木匠|厨子|掌柜|管家|师爷|门客|百姓|平民|路人|众人|大家|同门|世人|"
    "天下人|孤身|仙人|真君|圣主|堂主|庄主|店|铺|馆|楼|院|斋|轩|亭|台|"
    "山上|山下|水中|水里|空中|天上|地下|地上|面前|眼前|身后|身边|"
    "身旁|手边|床边|桌边|墙角|角落|一边|一遍|一句|一件|一柄|一把|一壶|"
    "一杯|一碗|一盘|一碟|一瓶|一坛|一箱|一柜|一床|一桌|一椅|一凳|一书|"
    "一卷|一页|一笔|一字|一句|一话|"
    "内城|宫殿内|山洞内|山门外|房屋内|马车内|道门|门外|屋内|房内|殿内|宫内|"
    "府内|山中|山下|山上|水上|空中|天上|地下|地上|城内|城外|界外|海外|国外|"
    "远方|远处|近处|身后|背后|面前|眼前|身边|周围|附近|旁边|左右|上下|前后|"
    "内外|中间|中央|中心|顶端|底部|上方|下方|内部|外部|表面|里面|外面|对面|"
    "这边|那边|领域|空间|世界|地方|区域|范围|周边|邻近|角落|尽头|深处|"
    "边缘|边界|门口|街头|巷尾|路边|桥头|河边|湖边|山顶|山腰|山脚|谷底|"
    "云端|天际|星空|宇宙|人间|凡间|世间|红尘|"
    # 高频通用名词（防止子-尾缀误入 motif 层）
    "孩子|日子|房子|屋子|桌子|椅子|杯子|袜子|鞋子|帽子|裤子|裙子|筷子|"
    "车子|样子|步子|口子|面子|底子|根子|头子|点子|袖子|绳子|带子|"
    "罐子|瓶子|盒子|袋子|架子|箱子|柜子|本子|册子|卷子|铺子|馆子|"
    "小伙子|小姑娘|大学生|高中生|初中生|小学生|同学|老师|班长|班主任|"
    "教室|学校|操场|宿舍|食堂|课本|试卷|作业|成绩|分数|考试|高考|"
    "手机|电脑|电视|空调|电灯|门窗|窗户|门口|马路|街道|公路|"
    "每天|每年|晚上|早上|中午|下午|夜里|周末|假期|暑假|寒假|"
    "爸妈|父母|爸爸|妈妈|父亲|母亲|爷爷|奶奶|外公|外婆|哥哥|姐姐|"
    "弟弟|妹妹|叔叔|阿姨|舅舅|舅妈|表哥|表姐|同桌|室友|邻居|"
    "生活|人生|未来|过去|现在|以前|之后|以后|之前|同时|"
    "厨房|卧室|卫生间|阳台|院子|走廊|楼梯|电梯|街角|桥边|树下|河边|湖边|"
    "朋友|同事|上司|老板|顾客|客人|一切|全部|整个|所有|许多|很多|很少|"
    "一点|一些|一个|一种|"
    # 通用高频非专名（古典/神魔文本：称谓/副词/动词短语，防"欢喜道/吩咐道"误入实体）
    "众僧|众官|众神|众猴|众妖|道人|道士|祖师|员外|先生|公公|婆婆|童子|童儿|"
    "老儿|老君|老妖|老魔|老怪|老龙|土地|菩萨|阎王|天王|王子|国丈|寿星|金星|"
    "星官|天尊|诸天|罗刹|金刚|雷公|宝殿|后边|房里|路旁|声高|心欢|不尽|"
    "欢喜|大怒|近前|吩咐|答应|迎接|高叫|仔细|传旨|礼拜|相迎|扯住|启奏|"
    "合掌|施礼|既如此|快早|莫要|莫念|有人|这等|又见|又问|王闻|吆喝|"
    "冷笑|厉声|古人|古人云|丞相|万岁|容易|路上|樵子|马匹|妖邪|魔王|云头|"
    "常言|常言道|方才|利害|只教|向前|胡说|莫胡说|开门|开了门|出门外|"
    "走出门|张开口|解了绳|收了法|捻着诀|念个咒|败了阵|伏在地|饶了他|"
    "管事的|须臾间|心惊|心中暗|多心经|到那里|往那里|看那里|都来|马来|"
    "闻得|闻得此|忽闻得|忍不住|"
    "这样|那样|这般|那般|如此|这厮|那厮|他们|她们|它们|我们|你们|咱们|"
    "人家|别人|"
    "厉声高|满心欢|多官|天师|国王|老者|罗汉|高老|"
    "诸将|魏主|蜀主|吴主|何必|何足|使者|回报|".split("|")
)

# ── 百家姓（含仙侠常见"玉/狐/琴"） + slice 补充姓氏 ──
_SURNAMES = frozenset(
    "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜戚谢"
    "邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳酆鲍史唐费廉岑薛"
    "雷贺倪汤滕殷罗毕郝邬安常乐于时傅皮卞齐康伍余元卜顾孟平黄和穆萧尹姚邵湛"
    "汪祁毛禹狄米贝明臧计伏成戴谈宋茅庞熊纪舒屈项祝董梁杜阮蓝闵席季麻强贾"
    "路娄危江童颜郭梅盛林刁钟徐邱骆高夏蔡田樊胡凌霍虞万支柯昝管卢莫经房裘"
    "缪干解应宗丁宣贲邓郁单杭洪包诸左石崔吉钮龚程嵇邢滑裴陆荣翁荀羊於惠甄"
    "曲家封芮羿储靳汲邴糜松井段富巫乌焦巴弓牧隗山谷车侯宓蓬全郗班仰秋仲伊"
    "宫宁仇栾暴甘钭厉戎祖武符刘景詹束龙叶幸司韶郜黎蓟溥印宿白怀蒲邰从鄂索"
    "咸籍赖卓蔺屠蒙池乔阴郁胥能苍双闻莘党翟谭贡劳逄姬申扶堵冉宰郦雍却璩桑"
    "桂濮牛寿通边扈燕冀郏浦尚农温别庄晏柴瞿阎充慕连茹习宦艾鱼容向古易慎戈"
    "廖庾终暨居衡步都耿满弘匡国文寇广禄阙东欧殳沃利蔚越夔隆师巩厩聂晁勾敖"
    "融冷訾辛阚那简饶空曾毋沙乜养鞠须丰巢关蒯相查后荆红游竺权逯盖益桓公玉"
    "狐琴叶钟"
)

# 称谓主体头字过滤（"拜观音/的长老"类动宾/虚词碎片不作称谓主体；
# 王母娘娘→母娘娘 碎片跳过依赖其左移主体头字为正常人名头）
_TITLE_BAD_HEAD = frozenset(
    "拜见请谢骂喊唤问求随跟送迎往向自至到为是有在的了那这将把被让对与和同"
    "亲朝谒参奉祭跪告别离依仗借拿抬指看说听闻这那其每各几"
)

# ── 势力/地名后缀（宗教府宫阁山海外岛洞城省域界门派观殿洲朝国郡…） ──
_PLACE_SUFFIXES = frozenset(
    "宗教府宫阁山海外岛洞城省域界门派观殿洲朝国郡州池泉谷林峰陵河江湖洋"
)
# ── 功法/器物后缀（功经术拳法诀印咒阵法丹药体器刀剑琴符旗钟鼎塔镜…） ──
_ARTIFACT_SUFFIXES = frozenset(
    "功经术拳法诀印咒阵法丹药体器刀剑琴符旗钟鼎塔镜笔册卷图盘珠环"
)

# ═══════════════════════════ 拉丁人名（西幻英文原文，P1） ═══════════════════════════

# 连续大写开头的英文词（单词级：Aragorn / Dorothy；词组级由调用方按序列合并）
_LATIN_NAME_RE = re.compile(r"(?<![A-Za-z])([A-Z][a-z]{2,})(?![A-Za-z])")
# 拉丁对话标签（"said Dorothy" / "Dorothy said" / "whispered the Wizard"）
_LATIN_DIALOG_BEFORE_RE = re.compile(
    r"\b(?:the\s+)?([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*)\s+"
    r"(said|asked|replied|cried|shouted|whispered|murmured|answered|added|agreed)\b"
)
_LATIN_DIALOG_AFTER_RE = re.compile(
    r"\b(said|asked|replied|cried|shouted|whispered|murmured|answered|added|agreed)\s+"
    r"(?:the\s+)?([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*)\b"
)
# 常见非专名大写词（句首/代词/连词），拉丁实体候选黑名单
_LATIN_STOPWORDS = frozenset(
    "The He She It They You We I A An And But Or For Nor So Yet Then When While"
    "As At In On Of To By With From Into Over Under About After Before During"
    "His Her Their Its Our Your My Me Him Them Us This That These Those There Here"
    "Was Were Is Are Be Been Being Do Does Did Done Not No Yes All Any Each Every"
    "One Two Three Four Five Six Seven Eight Nine Ten What Who Why How Where When"
    "Which Whom Once Now Again Also Only Just Very Much More Most Some Such Than"
    "Out Up Down Away Off Back Upon Against Through Among Within Without"
    "Chapter Chapter Chapters Sir Mr Mrs Ms Miss Dr Prof Lord Lady Master"
    "Father Mother Sister Brother Uncle Aunt Cousin Dear Good Great Little Old"
    "North South East West Up Down River City Town Land World King Queen Prince"
    "Duke Earl Lord Lady Knight Witch Dragon Castle".split()
)

# 中文音译间隔号名（哈利·波特 / 甘道夫·灰袍）。
# 左侧 ≤3 字、右侧 ≤2 字（主流音译名形态），不做后向边界（名字后常接动词）。
# 超长名（阿不思·邓布利多）依赖词频聚合或卡先验，README 注明局限。
_TRANSLIT_DOT_RE = re.compile(
    r"([\u4e00-\u9fff]{1,3}·[\u4e00-\u9fff]{1,2})"
)
_TRANSLIT_DOT_BOUNDED_RE = re.compile(
    r"(?<![\u4e00-\u9fff])([\u4e00-\u9fff]{1,3}·[\u4e00-\u9fff]{1,2})"
)


def discover_latin_entities(
    text: str,
    min_freq: int,
    *,
    chapters: dict[int, str] | None = None,
) -> set[str]:
    """拉丁字母人名/地名发现（西幻英文原文）。

    证据：大写开头单词 freq ≥ min_freq 且（对话标签语境 ≥2 或 跨章 span ≥3）；
    停用表过滤句首/代词/连词等非专名；词组（如 Minas Tirith）由对话标签
    或相邻序列补全为全名。
    """
    words: Counter[str] = Counter(_LATIN_NAME_RE.findall(text))
    entities: set[str] = set()

    # 对话标签两侧的名（“said Dorothy”“Dorothy said”）
    dialog_ctx: Counter[str] = Counter()
    for m in _LATIN_DIALOG_BEFORE_RE.finditer(text):
        dialog_ctx[m.group(1)] += 1
    for m in _LATIN_DIALOG_AFTER_RE.finditer(text):
        dialog_ctx[m.group(2)] += 1

    span_cache: dict[str, int] = {}

    def span(word: str) -> int:
        if word not in span_cache:
            span_cache[word] = _chapter_span(word, text, chapters)
        return span_cache[word]

    for word, freq in words.items():
        if word in _LATIN_STOPWORDS:
            continue
        if freq < min_freq:
            continue
        ctx = dialog_ctx.get(word, 0)
        # 对话标签语境 ≥1（"said Dorothy"）或 跨章 ≥3 即收；
        # 对话标签是强信号（专名出现在 said/asked 两侧）
        if ctx >= 1 or span(word) >= 3:
            entities.add(word)
    return entities

# ── 人物语境（第 3 证据：无姓氏高频人名须有"人味"出现） ──
# 通用化：并入古典说话动词（曰/对曰，三国/西游记类）与常见说话/动作动词
_PERSON_VERB_PATTERN = (
    r"说道|说|道|问|看着|抬起头|抬头|转身|走到|接过|递给|点头|摇头|"
    r"笑了|皱眉|心想|穿上|戴上|看向|望向|走进|离开|回到|坐在|站起|坐下|"
    r"开口|答道|回答|站起来|蹲下|伸手|放下|拿起|"
    r"曰|对曰|答曰|言曰|喝道|叫道|笑道|叹道|怒道|惊道|急道|喜道"
)
_PERSON_VERB_RE = re.compile(
    r"(?<![\u4e00-\u9fff])([\u4e00-\u9fff]{2,3})" rf"(?:{_PERSON_VERB_PATTERN})"
)


def _person_context_count(word: str, text: str) -> int:
    """统计候选词作为"人物动作主语"的出现次数（爱民说/爱民看向…）。

    注意：无前向边界——"甘道夫对阿拉贡说"中"阿拉贡"前是介词"对"，
    也应计入人物语境（P2 西幻/音译名场景）。
    """
    return len(re.findall(rf"{re.escape(word)}(?:{_PERSON_VERB_PATTERN})", text))


def complete_word_count(word: str, text: str) -> int:
    """完整词出现次数（前后均非汉字）——建卡提议的碎片过滤器。"""
    return len(
        re.findall(
            rf"(?<![\u4e00-\u9fff]){re.escape(word)}(?![\u4e00-\u9fff])", text
        )
    )


# ═══════════════════════════ FactProfile ═══════════════════════════

@dataclass
class FactProfile:
    """题材事实提取 profile：实体证据 + 谓词模板 + 线索层参数。"""

    id: str
    # 实体发现
    extra_surnames: frozenset = frozenset()
    high_freq_min: int = 20  # 无姓氏高频人名 freq 下限
    high_freq_span: int = 5  # 无姓氏高频人名跨章下限
    min_person_ctx: int = 2  # 无姓氏人名最低"人物语境"出现次数
    # 称谓后缀（profile 级，与全局 TITLE_SUFFIX 合并；西幻：骑士/法师/公爵…）
    title_suffixes: frozenset = frozenset()
    # 拉丁字母人名（西幻英文原文：Aragorn / Minas Tirith）
    latin_names: bool = False
    latin_min_freq: int = 12  # 拉丁人名词频下限
    latin_min_span: int = 3  # 拉丁人名跨章下限（或对话标签 ≥2 亦可）
    # motif 线索层
    motif_suffixes: frozenset = frozenset()
    min_motif_freq: int = 15
    min_motif_chapters: int = 5
    # 提取模板
    action_patterns: list = field(default_factory=list)  # [(regex, verb→predicate)]
    relationship_verbs: frozenset = frozenset()
    promise_keywords: tuple = ()  # 对话约定关键词
    max_facts: int = 8


# ── 动作模板（cultivation：事件流；动词具名捕获组） ──
_ACTION_RE = re.compile(
    r"(?P<a>[\u4e00-\u9fff]{2,4})(?P<v>杀死|击杀|救下|救出|击败|打败|抓住|放走|"
    r"斩杀|斩灭|镇压|追杀|护送|交给|送给|赠予|夺走|抢走|"
    r"娶了|收了|驯服|收服|控制|灭杀|救回|带进|带入|离开|加入|进入|"
    r"引兵|率军|率兵|遣使|遣|破|围|袭|斩|"
    r"打死|打伤|打退|杀败|擒住|擒获|捉住|困住|吓退|赶出|骗出|骗走|降服)"  # 通用打斗动词
    r"(?P<o>[\u4e00-\u9fff]{2,4})"
)
_ACTION_PREDICATE = {
    "杀死": "杀死", "击杀": "杀死", "斩杀": "杀死", "灭杀": "杀死", "斩": "斩",
    "打死": "打死", "打伤": "打伤",
    "救下": "救下", "救出": "救下", "救回": "救下",
    "击败": "击败", "打败": "击败", "打退": "打退", "杀败": "击败", "破": "攻破",
    "抓住": "抓住", "擒住": "抓住", "擒获": "抓住", "捉住": "抓住", "困住": "困住",
    "放走": "放走", "吓退": "吓退", "赶出": "赶出",
    "交给": "交给", "送给": "送给", "赠予": "交给",
    "夺走": "夺走", "抢走": "夺走", "骗出": "骗出", "骗走": "骗走",
    "离开": "离开", "加入": "加入", "进入": "进入",
    "引兵": "率军", "率军": "率军", "率兵": "率军", "遣使": "遣使", "遣": "遣",
    "围": "围攻", "袭": "袭击", "降服": "收服",
}

# ── 古典演义对话（文言"X曰"句式，三国演义等；宾语=引号内容，谓词"曰"） ──
_CLASSIC_SPEECH_RE = re.compile(
    r"(?P<a>[\u4e00-\u9fff]{2,4})(?P<v>曰|对曰|答曰|言曰)[：:]?(?P<o>“[^”]{0,20})"
)
_CLASSIC_SPEECH_PREDICATE = {
    "曰": "曰", "对曰": "曰", "答曰": "曰", "言曰": "曰",
}

# ── 通用说话模板（西游记/白话小说/网文："X道/说道/喝道…"；谓词"道"） ──
_GENERIC_SPEECH_RE = re.compile(
    r"(?P<a>[\u4e00-\u9fff]{2,4})"
    r"(?P<v>说道|答道|问道|言道|喝道|叫道|笑道|叹道|怒道|惊道|急道|喜道|"
    r"道|曰|对曰|答曰|言曰)[：:]?(?P<o>“[^”]{0,20})"
)
_GENERIC_SPEECH_PREDICATE = {
    "曰": "曰", "对曰": "曰", "答曰": "曰", "言曰": "曰",
    "道": "道", "说道": "道", "答道": "道", "问道": "道", "言道": "道",
    "喝道": "道", "叫道": "道", "笑道": "道", "叹道": "道", "怒道": "道",
    "惊道": "道", "急道": "道", "喜道": "道",
}

# ── slice 动作模板：赠送/信物/日常动作（动词在前，宾语 {2,8} 便于含量词） ──
_GIFT_RE = re.compile(
    r"(?P<a>[\u4e00-\u9fff]{2,4})(?P<v>送了一颗|送了一块|送了一包|送了一条|送了一个|送了一封|"
    r"送了他|送了她|递给|递了|送给|送了|给了|还给了|还给|收到|接过|捡起)"
    r"(?P<o>[\u4e00-\u9fff]{2,8})"
)
_WITH_RE = re.compile(
    r"(?P<a>[\u4e00-\u9fff]{2,4})(?P<v>系上|系了|戴上|戴了|带着|种下|留下|拿着|抱着|"
    r"握着|折了|织了|摘下|埋了|藏了|放进|取下|解下)(?P<o>[\u4e00-\u9fff]{2,4})"
)

# ── slice 关系推进 ──
_REL_RE = re.compile(
    r"([\u4e00-\u9fff]{2,4})(?:和|跟|与)([\u4e00-\u9fff]{2,4})"
    r"(牵手|拥抱|告白|确认|在一起|和好|分手|吵架|闹别扭|重归于好|订亲|结婚)"
)

# ── slice 对话约定（引号内承诺/等待） ──
_PROMISE_QUOTE_RE = re.compile(
    r"[“\"]([^”\"\n]*(?:约好|约定|答应|说好|承诺|等第[0-9零一二三四五六七八九十百]+次|"
    r"第[0-9零一二三四五六七八九十百]+次花开|七年|到时候|下次)[^”\"\n]{2,40})[”\"]"
)
_SPEAKER_RE = re.compile(r"([\u4e00-\u9fff]{2,4})(?:说道|说|道|问|开口|低声|笑道|回答|答道)")
PROMISE_PREDICATE = "约定"

_OWN_RE = re.compile(r"([\u4e00-\u9fff]{2,4})的([\u4e00-\u9fff]{2,4})")


def _mk_cultivation_profile() -> FactProfile:
    return FactProfile(
        id="cultivation",
        # 古典演义（第X回）适配：文言"X曰"对话 + 军事动词模板
        action_patterns=[
            (_ACTION_RE, _ACTION_PREDICATE),
            (_CLASSIC_SPEECH_RE, _CLASSIC_SPEECH_PREDICATE),
        ],
        relationship_verbs=frozenset(),
        promise_keywords=(),
    )


def _mk_generic_profile() -> FactProfile:
    """通用（未知文体）profile：白话/文言说话模板 + 通用事件流。

    面向未明确题材的长文：实体发现走共享四证据（含 曰/道 人物语境、
    停用字强证据绕行、组合名扩展），事实提取覆盖 "X道：“…”" 与
    通用打斗/交接动词——西游记（神魔话本）与网文白话均可命中。
    """
    return FactProfile(
        id="generic",
        # 古典称谓补充（神魔/仙侠：菩萨/佛祖/天王/魔王/大圣…）
        title_suffixes=frozenset(
            "菩萨|佛祖|天王|元帅|妖王|魔王|大仙|星君|老君|太岁|阎王|龙君|"
            "大圣|大王|罗汉|圣僧".split("|")
        ),
        action_patterns=[
            (_GENERIC_SPEECH_RE, _GENERIC_SPEECH_PREDICATE),
            (_ACTION_RE, _ACTION_PREDICATE),
            (_CLASSIC_SPEECH_RE, _CLASSIC_SPEECH_PREDICATE),
        ],
        relationship_verbs=frozenset(),
        promise_keywords=(),
    )


def _mk_slice_profile() -> FactProfile:
    return FactProfile(
        id="slice",
        # 日常文常见姓氏补充（爱/颜/唐/顾/沈/陆/祁/乔 等）
        extra_surnames=frozenset("爱颜唐顾沈陆祁乔白秦杜"),
        high_freq_min=20,
        high_freq_span=5,
        min_person_ctx=2,
        # 线索物件尾缀（罐/绳/树/花/糖/虫/瓶/盒/伞/笔/信/画/灯/鞋/纸/球/带/香/味）
        motif_suffixes=frozenset(
            "子绳树花糖虫瓶盒伞笔信画灯鞋纸球带香味罐结链坠符册页"
        ),
        min_motif_freq=15,
        min_motif_chapters=5,
        action_patterns=[
            (_ACTION_RE, _ACTION_PREDICATE),  # 事件流模板仍可用（救/杀/离开…）
            (_GIFT_RE, {"递给": "递给", "递了": "递给", "送给": "递给", "送了他": "递给",
                        "送了她": "递给", "送了一颗": "递给", "送了一块": "递给",
                        "送了一包": "递给", "送了一条": "递给", "送了一个": "递给",
                        "送了一封": "递给", "送了": "递给", "给了": "递给",
                        "还给了": "还给", "还给": "还给", "收到": "收到",
                        "接过": "接过", "捡起": "捡起"}),
            (_WITH_RE, {"系上": "系上", "系了": "系上", "戴上": "戴上", "戴了": "戴上",
                        "带着": "带着", "种下": "种下", "留下": "留下", "拿着": "拿着",
                        "抱着": "抱着", "握着": "握着", "折了": "折了", "织了": "织了",
                        "摘下": "摘下", "埋了": "埋下", "藏了": "藏起",
                        "放进": "放进"}),
        ],
        relationship_verbs=frozenset(
            "牵手|拥抱|告白|确认|在一起|和好|分手|吵架|闹别扭|重归于好|订亲|结婚".split("|")
        ),
        promise_keywords=("约好", "约定", "答应", "说好", "等第", "第七次", "七年"),
    )


# ── 西幻动作模板（P2：魔法/武器/王国事件；动词具名捕获组） ──
_WESTERN_ACTION_RE = re.compile(
    r"(?P<a>[\u4e00-\u9fff]{2,4})(?P<v>施展|释放|吟唱|咏唱|召唤|拔出|抽出|骑上|骑乘|"
    r"签订|缔结|立下|诅咒|赐福|复活|加冕|统帅|率领|点燃|夺取|占据|攻占|"
    r"守护|护送|救出|击败|杀死|斩杀)(?P<o>[\u4e00-\u9fff]{2,6})"
)
_WESTERN_ACTION_PREDICATE = {
    "施展": "施展", "释放": "施展", "吟唱": "吟唱", "咏唱": "吟唱",
    "召唤": "召唤", "拔出": "拔出", "抽出": "拔出",
    "骑上": "骑乘", "骑乘": "骑乘",
    "签订": "缔约", "缔结": "缔约", "立下": "立誓",
    "诅咒": "诅咒", "赐福": "赐福", "复活": "复活", "加冕": "加冕",
    "统帅": "率领", "率领": "率领",
    "点燃": "点燃", "夺取": "夺取", "占据": "占据", "攻占": "攻占",
    "守护": "守护", "护送": "护送", "救出": "救下",
    "击败": "击败", "杀死": "杀死", "斩杀": "杀死",
}


def _mk_western_profile() -> FactProfile:
    """西幻（西方玄幻/奇幻）：拉丁人名 + 中文音译名 + 魔法/王国事件流。"""
    return FactProfile(
        id="western",
        latin_names=True,
        latin_min_freq=12,
        # 西幻称谓后缀（骑士/法师/公爵…，并入全局称谓证据）
        title_suffixes=frozenset(
            "骑士|法师|大法师|魔导师|牧师|祭司|主教|圣女|圣子|巫师|术士|贤者|先知|"
            "爵士|公爵|伯爵|侯爵|子爵|男爵|领主|女王|国王|王子|公主|船长|元帅|将军"
            .split("|")
        ),
        action_patterns=[
            (_WESTERN_ACTION_RE, _WESTERN_ACTION_PREDICATE),
            (_ACTION_RE, _ACTION_PREDICATE),  # 通用事件流兜底
        ],
        relationship_verbs=frozenset(
            "爱上|订婚|背叛|效忠|结盟|决裂|宣战|和解|复国|复仇".split("|")
        ),
        promise_keywords=("发誓", "起誓", "诅咒", "预言", "誓约", "承诺", "约定"),
    )


CULTIVATION_PROFILE = _mk_cultivation_profile()
SLICE_PROFILE = _mk_slice_profile()
WESTERN_PROFILE = _mk_western_profile()
GENERIC_PROFILE = _mk_generic_profile()
PROFILES: dict[str, FactProfile] = {
    "cultivation": CULTIVATION_PROFILE,
    "slice": SLICE_PROFILE,
    "western": WESTERN_PROFILE,
    "generic": GENERIC_PROFILE,
}


def get_profile(profile_id: str | None) -> FactProfile:
    """取题材 profile；未知 id 回退通用 generic（未知文体默认）。"""
    return PROFILES.get(profile_id or "generic", GENERIC_PROFILE)


# ═══════════════════════════ 实体发现 ═══════════════════════════

@dataclass
class EntityDiscovery:
    """实体发现结果：entities（可作主语）/ motifs（线索物件，仅作宾语）。"""

    entities: set[str] = field(default_factory=set)
    motifs: set[str] = field(default_factory=set)


def _chapter_span(word: str, text: str, chapters: dict[int, str] | None) -> int:
    """跨章跨度：有命中的章节数（无章节信息时按 3000 字块近似）。"""
    if chapters is not None:
        return sum(1 for ch in chapters.values() if word in ch)
    # 块近似：把 text 切成 3000 字块统计命中块数
    span = 0
    for i in range(0, len(text), 3000):
        if word in text[i : i + 3000]:
            span += 1
    return span


def discover_entities(
    text: str,
    min_freq: int = 8,
    *,
    profile: FactProfile | None = None,
    card_names: set[str] | None = None,
    item_names: set[str] | None = None,
    chapters: dict[int, str] | None = None,
) -> EntityDiscovery:
    """从全文发现实体集（四证据并集）+ motif 线索层（卡名先验金标准优先）。

    - 卡名/物品名（作者确认过的资产）直接并入，绕过统计证据。
    - 第 3 证据（高频跨章人名）不要求姓氏，但要求跨章跨度 +
      人物语境出现次数（爱民 1006 次/27 章/语境充分 → 必中）。
    """
    profile = profile or CULTIVATION_PROFILE
    entities: set[str] = set(card_names or [])
    motifs: set[str] = set(item_names or [])

    # 证据 1：称谓后缀（高置信，频率 + 主体过滤去噪）
    # P1: profile 级称谓合并（西幻：骑士/法师/公爵…；通用：菩萨/大圣/天王…）
    title_suffix = TITLE_SUFFIX
    if profile.title_suffixes:
        title_suffix = f"{TITLE_SUFFIX}|{'|'.join(sorted(profile.title_suffixes))}"
    # 主体非贪婪捕获 + 后缀分组（profile 级后缀不计入 _title_len，
    # 直接用捕获的后缀截取主体——修复 profile 后缀词 body 恒为空的问题）。
    # 无前向边界：中嵌语段的称谓（拜观音菩萨/那太上老君）也计入；
    # 头字为动/虚词的外层匹配（拜观音）向内层探测真实称谓。
    title_re = re.compile(
        rf"([\u4e00-\u9fff]{{2,4}}?)((?:{title_suffix}))"
    )
    title_counter: Counter[str] = Counter()
    _title_exempt = ("太上", "太乙", "太白")
    for m in title_re.finditer(text):
        mm = m
        while True:
            body = mm.group(1)
            if body[0] not in _TITLE_BAD_HEAD and (
                body[0] not in _STOP_CHARS or body in _title_exempt
            ) and (body[-1] not in _STOP_CHARS or body in _title_exempt):
                break
            nxt = title_re.match(text, mm.start() + 1)
            if nxt is None or nxt.start() != mm.start() + 1:
                break
            mm = nxt
        body, name = mm.group(1), mm.group(0)
        # 主体头字过滤（拜观音/的长老）+ 首/尾停用字（古典尊号头豁免）
        if body and body[0] not in _TITLE_BAD_HEAD \
                and (body[0] not in _STOP_CHARS or body in _title_exempt) \
                and (body[-1] not in _STOP_CHARS or body in _title_exempt):
            title_counter[name] += 1
    for name, freq in title_counter.items():
        if freq >= min_freq:
            entities.add(name)

    counter: Counter[str] = Counter(_OVERLAP_2.findall(text))
    counter.update(_OVERLAP_3.findall(text))
    bounded: Counter[str] = Counter(_WORD_BOUNDED.findall(text))
    left_bounded: Counter[str] = Counter(_LEFT_BOUNDED_RE.findall(text))
    surnames = _SURNAMES | profile.extra_surnames
    motif_suffixes = profile.motif_suffixes
    # motif 跨章阈值随章节数缩放：长书保持 5 章，短书按 1/5 章节数降档
    n_chapters = len(chapters) if chapters else 5
    motif_span_need = min(profile.min_motif_chapters, max(1, n_chapters // 5))

    # 证据 3 候选（freq/跨章先筛）、2 字姓氏词 bounded 兜底（唐僧类：
    # "唐僧道/云长曰"中名字恒嵌语段内，整词边界天然偏低 → 用语境兜底）、
    # 强证据绕行候选（如来类：首停用+尾动词）。
    ev3_need = max(profile.high_freq_min, 5 * min_freq)
    ctx_need: dict[str, int] = {}
    surname_ctx_need: dict[str, int] = {}
    strong3_words: set[str] = set()

    for word, freq in counter.items():
        if word in entities or word in motifs:
            continue
        if word in _EXCLUDE_WORDS:
            continue

        # motif 线索层优先判定（宽松门槛：词频 + 跨章 + 线索尾缀 + 左边界出现）。
        # 放在姓氏证据之前，避免"红绳(红姓)/薄荷糖(薄姓)"被吸入人物实体；
        # 左边界 ≥1 过滤重叠窗口碎片（璃罐/火虫/荷糖 词首前恒为汉字）。
        if (
            motif_suffixes
            and freq >= profile.min_motif_freq
            and _chapter_span(word, text, chapters) >= motif_span_need
            and word[-1] in motif_suffixes
            and left_bounded.get(word, 0) >= 1
        ):
            motifs.add(word)
            continue

        threshold = 3 * min_freq if len(word) == 2 else min_freq
        if freq < threshold:
            continue
        bounded_need = 0 if min_freq < 3 else max(1, min(3, freq // 5))

        # 首/尾停用字与尾部动词过滤（"唐僧道/高叫/他们"类）；
        # 仅白名单首停用字的强证据词（如/哪/太：如来/哪吒/太宗）进入绕行候选池
        if (
            word[0] in _STOP_CHARS
            or word[-1] in _STOP_CHARS
            or word[-1] in _TAIL_VERBS
        ):
            if (
                word[0] in _STRONG3_HEAD_CHARS
                and freq >= ev3_need
                and _chapter_span(word, text, chapters) >= profile.high_freq_span
            ):
                strong3_words.add(word)
                ctx_need[word] = max(profile.min_person_ctx, freq // 10)
            continue

        # 证据 2：姓氏开头（人名）或 势力/功法后缀（地名/器物）
        if word[0] in surnames:
            if bounded.get(word, 0) < bounded_need:
                # 整词边界不足 → 2-3 字词允许人物语境兜底（唐僧/李天王类）
                if len(word) in (2, 3):
                    surname_ctx_need[word] = profile.min_person_ctx
                continue
            entities.add(word)
            continue
        if word[-1] in _PLACE_SUFFIXES or word[-1] in _ARTIFACT_SUFFIXES:
            if bounded.get(word, 0) >= bounded_need:
                entities.add(word)
            continue

        # 证据 3 候选：高频 + 跨章（人物语境批量计算在循环外）
        if freq >= ev3_need and _chapter_span(word, text, chapters) >= profile.high_freq_span:
            ctx_need[word] = max(profile.min_person_ctx, freq // 10)

    # 批量人物语境：无边界（ctx）与左边界（left_ctx）一次全文扫描
    if ctx_need or surname_ctx_need:
        pool = sorted(set(ctx_need) | set(surname_ctx_need), key=len, reverse=True)
        alt = "|".join(re.escape(w) for w in pool)
        ctx_counter: Counter[str] = Counter(
            m.group(1) for m in re.finditer(rf"({alt})(?:{_PERSON_VERB_PATTERN})", text)
        )
        left_ctx_counter: Counter[str] = Counter(
            m.group(1)
            for m in re.finditer(rf"(?<![\u4e00-\u9fff])({alt})(?:{_PERSON_VERB_PATTERN})", text)
        )
    else:
        ctx_counter, left_ctx_counter = Counter(), Counter()

    # 姓氏词语境兜底（须左边界语境且尾字非动词/虚词：
    # 挡住"冷笑道/王闻言/孔明遂"类副词与动词短语）
    for word, need in surname_ctx_need.items():
        if (
            ctx_counter.get(word, 0) >= need
            and left_ctx_counter.get(word, 0) >= 2
            and word[-1] not in _EXPAND_BAD_TAIL
        ):
            entities.add(word)

    # 证据 3：高频 + 跨章 + 人物语境密度 + 左边界语境
    # （"悟空道"多现于句首/标点后；"欢喜/仔细"等形容词恒嵌语段内被过滤）
    for word, need in ctx_need.items():
        ctx = ctx_counter.get(word, 0)
        if ctx < need:
            continue
        if word in strong3_words:
            # 强证据绕行（哪吒/太宗/如来类：首停用+左边界语境 ≥2）
            if left_ctx_counter.get(word, 0) >= 2:
                entities.add(word)
            continue
        left_need = max(2, counter[word] // 15) if min_freq >= 3 else 0
        if left_ctx_counter.get(word, 0) >= left_need:
            entities.add(word)

    # 组合名扩展：已发现实体 X 的扩展形式（X八戒/X三藏/X大圣）——
    # 3-4 字、含实体子串、词频达标、整词边界 ≥1、首字非动词/形容词/停用、
    # 尾字非说话/动作动词（"悟空道/见八戒/好行者"类碎片被首尾过滤）。
    for word in sorted(counter, key=lambda w: -counter[w]):
        if len(word) not in (3, 4) or word in entities or word in motifs:
            continue
        if word in _EXCLUDE_WORDS:
            continue
        if counter[word] < min_freq or left_bounded.get(word, 0) < 1:
            continue
        # 整词边界 ≥1（"保唐僧/假唐僧"类动宾碎片无整词出现）；
        # 姓氏头豁免（牛魔王/唐三藏 恒嵌"X道"语段内，bounded 天然为 0）
        if bounded.get(word, 0) < 1 and word[0] not in surnames:
            continue
        if word[0] in _STOP_CHARS or word[0] in _EXPAND_BAD_HEAD:
            continue
        if word[-1] in _STOP_CHARS or word[-1] in _EXPAND_BAD_TAIL:
            continue
        if any(len(e) >= 2 and e in word for e in entities):
            entities.add(word)
            continue

    # P1: 中文音译间隔号名（哈利·波特）：双轨频率（边界版高置信 / 宽松版防粘连）
    bounded_translit = Counter(_TRANSLIT_DOT_BOUNDED_RE.findall(text))
    raw_translit = Counter(_TRANSLIT_DOT_RE.findall(text))
    for name, freq in bounded_translit.items():
        if freq >= max(2, min_freq // 2):
            entities.add(name)
    for name, freq in raw_translit.items():
        if any(n != name and name in n for n in bounded_translit):
            continue  # 介词粘连变体（"对哈利·波特说" ⊃ "哈利·波特"）丢弃
        if freq >= max(3, min_freq):
            entities.add(name)

    # P1: 拉丁人名（西幻英文原文，profile.latin_names 时启用）
    if profile.latin_names:
        # 阈值随文本规模联动：短文本/测试场景用 min_freq 推导，长书用 latin_min_freq
        latin_min = min(profile.latin_min_freq, max(3, min_freq))
        latin = discover_latin_entities(
            text, latin_min, chapters=chapters
        )
        entities |= latin

    return EntityDiscovery(entities=entities, motifs=motifs)


def _title_len(name: str) -> int:
    """返回称谓后缀的长度（贪心匹配最长后缀）。"""
    for suffix in sorted(TITLE_SUFFIX.split("|"), key=len, reverse=True):
        if name.endswith(suffix):
            return len(suffix)
    return 0


# ═══════════════════════════ 事实提取 ═══════════════════════════

def extract_heuristic_facts(
    content: str,
    chapter_no: int,
    entities: set[str],
    motifs: set[str] | None = None,
    profile: FactProfile | None = None,
) -> list[FactTriple]:
    """实体驱动的事实提取。主语必须 ∈ entities；宾语 ∈ entities∪motifs。

    优先级：关系/对话约定 > 动作 > 拥有；"登场"已移除（信息量低且挤压名额）。
    P2: western 动作谓词的宾语放宽（巨鹰/咒语/战马等普通名词可作宾语，
    主语仍严格 ∈ 实体，避免主语垃圾化）。
    """
    profile = profile or CULTIVATION_PROFILE
    motifs = motifs or set()
    objects_allowed: set[str] = entities | motifs
    # P2: western 宽松宾语谓词（动作模板产物；量词清洗见 _clean_obj）
    relaxed_predicates = (
        set(_WESTERN_ACTION_PREDICATE.values()) if profile.id == "western" else set()
    )
    facts: list[FactTriple] = []
    seen: set[tuple[str, str, str]] = set()
    max_facts = profile.max_facts

    def add(subject: str, predicate: str, obj: str) -> bool:
        key = (subject, predicate, obj)
        if key in seen or len(facts) >= max_facts:
            return False
        if subject not in entities:
            return False
        # 约定/文言"曰"/通用"道"/西幻宽松宾语 允许非实体宾语，其余须 ∈ 实体∪motif
        if (
            predicate not in (PROMISE_PREDICATE, "曰", "道")
            and predicate not in relaxed_predicates
            and obj not in objects_allowed
        ):
            return False
        seen.add(key)
        facts.append(
            FactTriple(subject=subject, predicate=predicate, object=obj, chapter_no=chapter_no)
        )
        return True

    # ── 关系推进（slice）：X和Y牵手/告白/和好… ──
    if profile.relationship_verbs:
        for m in _REL_RE.finditer(content):
            a, b, verb = m.group(1), m.group(2), m.group(3)
            if verb in profile.relationship_verbs:
                add(a, verb, b)

    # ── 对话约定（slice）：引号内承诺 → 回溯说话人 ──
    if profile.promise_keywords:
        for m in _PROMISE_QUOTE_RE.finditer(content):
            quote = m.group(1)
            if not any(kw in quote for kw in profile.promise_keywords):
                continue
            speaker = _find_speaker(content, m.start(), entities)
            if speaker is not None:
                add(speaker, PROMISE_PREDICATE, quote[:24])

    # ── 动作模板（事件流 + slice 日常动作） ──
    for pattern, predicate_map in profile.action_patterns:
        for m in pattern.finditer(content):
            a = m.group("a")
            predicate = predicate_map.get(m.group("v"), "动作")
            captured = m.group("o")
            # 说话模板（曰/道）：宾语直接取引号内容（主语已在实体集内）；
            # 不走实体候选（"悟空道：“八戒…”"的宾语是话语而非八戒）
            if predicate in ("曰", "道"):
                obj = _clean_quote(captured)
                if obj:
                    add(a, predicate, obj)
                continue
            # 宾语候选：捕获文本中出现的实体/motif，取最后出现者（信物/目标）
            cands: list[tuple[int, str]] = []
            for w in entities:
                pos = captured.find(w)
                if pos >= 0:
                    cands.append((pos, w))
            for w in motifs:
                pos = captured.find(w)
                if pos >= 0:
                    cands.append((pos, w))
            if cands:
                cands.sort(key=lambda t: t[0])
                add(a, predicate, cands[-1][1])
            elif predicate in relaxed_predicates:
                # P2: western 宽松宾语（巨鹰/咒语/战马等普通名词），量词清洗
                obj = _clean_obj(captured)
                if obj:
                    add(a, predicate, obj)

    # ── 拥有：实体A 的 实体B（双实体约束） ──
    for m in _OWN_RE.finditer(content):
        a, b = m.group(1), m.group(2)
        if a in entities and b in entities:
            add(a, "拥有", b)

    return facts


# ── P2: western 宽松宾语清洗（去量词/代词前缀） ──
_OBJ_PREFIX_RE = re.compile(
    r"^(?:一只|一个|一头|一匹|一名|一位|一柄|一把|一件|一切|所有|许多|"
    r"他|她|它|我|你|这|那|他的|她的|我的|你的|他们的|她们|它们的)"
)
_OBJ_TAIL_RE = re.compile(r"[的了着]$")


def _clean_obj(raw: str) -> str:
    """动作宾语去量词/代词前缀与语气词尾；剩余 ≥2 字才返回。"""
    s = _OBJ_PREFIX_RE.sub("", raw.strip())
    s = _OBJ_TAIL_RE.sub("", s)
    return s if len(s) >= 2 else ""


_QUOTE_TAIL_RE = re.compile(r"[\s，。；：！？、…—]+$")


def _clean_quote(raw: str) -> str:
    """说话模板宾语：去开引号与尾部标点；剩余 ≥2 字才返回。"""
    s = raw.strip().lstrip("\u201c\u201d\u300c\u300d\"'")
    s = _QUOTE_TAIL_RE.sub("", s)
    return s if len(s) >= 2 else ""


def _find_speaker(content: str, quote_start: int, entities: set[str]) -> str | None:
    """回溯引号前 80 字内的最近说话人；须 ∈ 实体集。

    P1: 同时支持中文（X说道）与英文（X said / said X）说话标签。
    """
    window = content[max(0, quote_start - 80) : quote_start]
    best: str | None = None
    best_pos = -1
    for m in _SPEAKER_RE.finditer(window):
        cand = m.group(1)
        if cand in entities and m.start() > best_pos:
            best, best_pos = cand, m.start()
    for m in _LATIN_DIALOG_BEFORE_RE.finditer(window):
        cand = m.group(1)
        if cand in entities and m.start() > best_pos:
            best, best_pos = cand, m.start()
    for m in _LATIN_DIALOG_AFTER_RE.finditer(window):
        cand = m.group(2)
        if cand in entities and m.start() > best_pos:
            best, best_pos = cand, m.start()
    return best
