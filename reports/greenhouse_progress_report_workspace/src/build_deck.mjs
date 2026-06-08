import fs from "node:fs/promises";
import path from "node:path";
import {
  Presentation,
  PresentationFile,
  row,
  column,
  grid,
  layers,
  panel,
  text,
  shape,
  rule,
  fill,
  hug,
  fixed,
  wrap,
  grow,
  fr,
  auto,
} from "@oai/artifact-tool";

const OUT = {
  workspacePptx: "output/output.pptx",
  finalPptx: "../greenhouse_progress_report_20260430.pptx",
  previewDir: "scratch/previews",
};

const W = 1920;
const H = 1080;
const colors = {
  bg: "#F6FAF4",
  bg2: "#EEF7EF",
  ink: "#17352E",
  muted: "#5E756F",
  green: "#2E7D5B",
  green2: "#57A773",
  mint: "#DDEFE5",
  mint2: "#C6E6D3",
  blue: "#2F6F9F",
  blue2: "#D9EAF5",
  amber: "#B7791F",
  amber2: "#F4E2BD",
  red: "#B4443F",
  red2: "#F2D8D5",
  white: "#FFFFFF",
  line: "#C9D8CF",
  dark: "#0F2B24",
};

const titleStyle = { fontSize: 47, bold: true, color: colors.ink, lineSpacing: 1.08 };
const subtitleStyle = { fontSize: 24, color: colors.muted, lineSpacing: 1.24 };
const bodyStyle = { fontSize: 24, color: colors.ink, lineSpacing: 1.22 };
const smallStyle = { fontSize: 17, color: colors.muted, lineSpacing: 1.18 };
const labelStyle = { fontSize: 18, color: colors.green, bold: true };

const presentation = Presentation.create({ slideSize: { width: W, height: H } });

function t(value, opts = {}) {
  return text(value, {
    width: opts.width ?? fill,
    height: opts.height ?? hug,
    style: { ...bodyStyle, ...(opts.style || {}) },
    name: opts.name,
    columnSpan: opts.columnSpan,
    rowSpan: opts.rowSpan,
  });
}

function chip(label, tone = "green", width = hug) {
  const fillColor = tone === "red" ? colors.red2 : tone === "amber" ? colors.amber2 : tone === "blue" ? colors.blue2 : colors.mint;
  const textColor = tone === "red" ? colors.red : tone === "amber" ? colors.amber : tone === "blue" ? colors.blue : colors.green;
  return panel(
    {
      width,
      height: hug,
      padding: { x: 18, y: 8 },
      fill: fillColor,
      borderRadius: 18,
    },
    t(label, { style: { fontSize: 17, bold: true, color: textColor, alignment: "center" } }),
  );
}

function bulletList(items, opts = {}) {
  return column(
    { width: opts.width ?? fill, height: hug, gap: opts.gap ?? 14 },
    items.map((item, idx) =>
      row({ width: fill, height: hug, gap: 14, align: "start" }, [
        t("•", { width: fixed(22), style: { fontSize: opts.dotSize ?? 24, color: opts.dotColor ?? colors.green, bold: true } }),
        t(item, { width: fill, style: { fontSize: opts.fontSize ?? 23, color: opts.color ?? colors.ink, lineSpacing: 1.2 } }),
      ]),
    ),
  );
}

function metric(label, value, caption, tone = "green") {
  const color = tone === "red" ? colors.red : tone === "amber" ? colors.amber : tone === "blue" ? colors.blue : colors.green;
  return column({ width: fill, height: hug, gap: 6 }, [
    t(label, { style: { fontSize: 17, color: colors.muted, bold: true } }),
    t(value, { style: { fontSize: 40, color, bold: true, lineSpacing: 1.0 } }),
    t(caption, { style: { fontSize: 16, color: colors.muted, lineSpacing: 1.12 } }),
  ]);
}

function stepBox(num, heading, detail, tone = "green") {
  const fillColor = tone === "blue" ? colors.blue2 : tone === "amber" ? colors.amber2 : tone === "red" ? colors.red2 : colors.mint;
  const accent = tone === "blue" ? colors.blue : tone === "amber" ? colors.amber : tone === "red" ? colors.red : colors.green;
  return panel(
    {
      width: fill,
      height: fixed(142),
      padding: { x: 22, y: 18 },
      fill: fillColor,
      borderRadius: 18,
    },
    row({ width: fill, height: fill, gap: 14, align: "start" }, [
      panel(
        { width: fixed(42), height: fixed(42), fill: colors.white, borderRadius: 21 },
        t(num, { style: { fontSize: 18, bold: true, color: accent, alignment: "center" } }),
      ),
      column({ width: fill, height: hug, gap: 8 }, [
        t(heading, { style: { fontSize: 22, bold: true, color: colors.ink } }),
        t(detail, { style: { fontSize: 17, color: colors.muted, lineSpacing: 1.16 } }),
      ]),
    ]),
  );
}

function moduleBox(title, body, tone = "green", height = 152) {
  const fillColor = tone === "blue" ? "#EEF6FB" : tone === "amber" ? "#FBF4E8" : tone === "red" ? "#FBEEEE" : colors.white;
  const accent = tone === "blue" ? colors.blue : tone === "amber" ? colors.amber : tone === "red" ? colors.red : colors.green;
  return panel(
    {
      width: fill,
      height: fixed(height),
      padding: { x: 24, y: 18 },
      fill: fillColor,
      line: { color: colors.line, weight: 1 },
      borderRadius: 16,
    },
    column({ width: fill, height: hug, gap: 10 }, [
      row({ width: fill, height: hug, gap: 12, align: "center" }, [
        shape({ width: fixed(10), height: fixed(28), fill: accent, borderRadius: 5 }),
        t(title, { style: { fontSize: 23, bold: true, color: colors.ink } }),
      ]),
      t(body, { style: { fontSize: 18, color: colors.muted, lineSpacing: 1.18 } }),
    ]),
  );
}

function header(slideNo, title, subtitle) {
  return column({ width: fill, height: hug, gap: 10 }, [
    row({ width: fill, height: hug, align: "center" }, [
      t(`0${slideNo}`.slice(-2), { width: fixed(58), style: { fontSize: 19, bold: true, color: colors.green } }),
      rule({ width: fixed(90), stroke: colors.green, weight: 3 }),
      t("温室控制项目进度汇报", { width: fill, style: { fontSize: 17, color: colors.muted, alignment: "right" } }),
    ]),
    t(title, { name: `slide-${slideNo}-title`, style: titleStyle }),
    subtitle ? t(subtitle, { style: subtitleStyle, width: wrap(1420) }) : null,
  ].filter(Boolean));
}

function addSlide(slideNo, title, subtitle, bodyNode, notes) {
  const slide = presentation.slides.add();
  slide.compose(
    layers({ width: fill, height: fill }, [
      shape({ width: fill, height: fill, fill: colors.bg }),
      shape({ width: fixed(460), height: fixed(1080), fill: slideNo % 2 === 0 ? "#EDF7F0" : "#F2F8F2" }),
      column(
        { width: fill, height: fill, padding: { x: 84, y: 58 }, gap: 28 },
        [
          header(slideNo, title, subtitle),
          bodyNode,
          row({ width: fill, height: hug, align: "center" }, [
            t("数据来源：项目文档与 frozen benchmark / sentinel gate 诊断输出", { width: fill, style: { fontSize: 13, color: "#789089" } }),
            t(`${slideNo}/12`, { width: fixed(74), style: { fontSize: 13, color: "#789089", alignment: "right" } }),
          ]),
        ],
      ),
    ]),
    { frame: { left: 0, top: 0, width: W, height: H }, baseUnit: 8 },
  );
  slide.speakerNotes.setText(notes);
  return slide;
}

function addCover() {
  const slide = presentation.slides.add();
  slide.compose(
    layers({ width: fill, height: fill }, [
      shape({ width: fill, height: fill, fill: "#0F2B24" }),
      shape({ width: fixed(560), height: fill, fill: "#173D32" }),
      shape({ width: fixed(1920), height: fixed(130), fill: "#2E7D5B" }),
      column(
        { width: fill, height: fill, padding: { x: 100, y: 74 }, gap: 36 },
        [
          row({ width: fill, height: hug, align: "center" }, [
            chip("LLM-RSPC", "green", fixed(142)),
            chip("Frozen Replay", "blue", fixed(176)),
            chip("Tomato Safety", "amber", fixed(178)),
          ]),
          column({ width: wrap(1240), height: hug, gap: 24 }, [
            t("温室控制项目阶段进展", { style: { fontSize: 76, bold: true, color: colors.white, lineSpacing: 1.02 } }),
            t("LLM 引导的鲁棒设定点规划与安全滚动控制", {
              style: { fontSize: 34, color: "#CFE6D8", lineSpacing: 1.12 },
              width: wrap(1180),
            }),
          ]),
          row({ width: fill, height: fixed(260), gap: 42, align: "end" }, [
            metric("当前主线", "LLM-RSPC", "高层计划 + 低层安全执行", "green"),
            metric("验证体系", "Cache Replay", "冻结 LLM 输出，比较控制器本体", "blue"),
            metric("最新发现", "Sentinel Gate", "跨季节检查拦截高温泛化风险", "amber"),
          ]),
          t("面向老师/课题组的阶段汇报 · 2026-04-30", {
            style: { fontSize: 21, color: "#A9C7BA" },
          }),
        ],
      ),
    ]),
    { frame: { left: 0, top: 0, width: W, height: H }, baseUnit: 8 },
  );
  slide.speakerNotes.setText("开场先说明：本次汇报不是宣称最终性能已经最好，而是说明目前系统流程、已完成模块、验证发现的问题，以及下一步如何稳健推进。");
}

function simpleTable(headers, rows, widths) {
  const allRows = [
    row({ width: fill, height: fixed(54), gap: 0 }, headers.map((h, i) =>
      panel({ width: widths[i], height: fill, fill: colors.green, padding: { x: 14, y: 12 } },
        t(h, { style: { fontSize: 17, bold: true, color: colors.white, alignment: i === 0 ? "left" : "center" } })),
    )),
    ...rows.map((r, idx) =>
      row({ width: fill, height: fixed(62), gap: 0 }, r.map((c, i) =>
        panel(
          {
            width: widths[i],
            height: fill,
            fill: idx % 2 === 0 ? colors.white : "#F1F7F2",
            line: { color: "#E0E9E3", weight: 1 },
            padding: { x: 14, y: 12 },
          },
          t(c, { style: { fontSize: i === 0 ? 16 : 17, color: i === r.length - 1 ? colors.green : colors.ink, bold: i === r.length - 1, alignment: i === 0 ? "left" : "center" } }),
        ),
      )),
    ),
  ];
  return column({ width: fill, height: hug, gap: 0 }, allRows);
}

addCover();

addSlide(
  2,
  "研究目标：不是让 LLM 直接控温室，而是让它参与可验证决策",
  "当前方法定位为 LLM-guided Robust Setpoint Planning and Safe Rollout Control。",
  grid(
    { width: fill, height: grow(1), columns: [fr(1), fr(1)], rows: [auto, auto], columnGap: 28, rowGap: 24 },
    [
      moduleBox("安全约束", "温度、RH、VPD、露点裕度和执行器冲突必须显式可诊断，而不是只靠 reward。", "green", 176),
      moduleBox("经济目标", "降低加热、CO2、补光和无效 heat+vent 冲突，同时保持作物状态安全。", "blue", 176),
      moduleBox("可解释过程", "保留状态分析、重规划原因、LLM 原始计划、合同修正、fallback 和 guardrail 记录。", "amber", 176),
      moduleBox("可复现实验", "用 plan cache 冻结高层 LLM 计划，让 RSPC/HEM/MC-SERO 消融在同一计划下比较。", "green", 176),
    ],
  ),
  "这一页先定研究目标：LLM 不直接输出最终动作并无约束执行，而是承担高层计划与专家知识注入；真正闭环安全由 RSPC、fallback、guardrail 和可复现评估体系保证。",
);

addSlide(
  3,
  "系统总流程：高层计划低频生成，底层控制每步安全滚动",
  "整体闭环分为状态分析、LLM 规划、计划合同、候选动作、护栏过滤和环境执行。",
  column({ width: fill, height: grow(1), gap: 26 }, [
    grid(
      { width: fill, height: hug, columns: [fr(1), auto, fr(1), auto, fr(1)], columnGap: 14, alignItems: "center" },
      [
        stepBox("1", "状态抓取与风险分析", "读取 T/RH/VPD/CO2/天气/作物状态，判断是否触发重规划。"),
        t("→", { width: fixed(42), style: { fontSize: 35, color: colors.green, bold: true, alignment: "center" } }),
        stepBox("2", "LLM 高层规划", "生成锚点动作、未来目标状态和短期策略意图。", "blue"),
        t("→", { width: fixed(42), style: { fontSize: 35, color: colors.green, bold: true, alignment: "center" } }),
        stepBox("3", "Setpoint Contract", "补齐/裁剪目标温室状态，保证目标轨迹可执行。", "amber"),
      ],
    ),
    grid(
      { width: fill, height: hug, columns: [fr(1), auto, fr(1), auto, fr(1)], columnGap: 14, alignItems: "center" },
      [
        stepBox("4", "RSPC Rollout", "读取计划，与锚点动作加权，误差修正后生成候选控制", "blue"),
        t("→", { width: fixed(42), style: { fontSize: 35, color: colors.green, bold: true, alignment: "center" } }),
        stepBox("5", "Fallback + Guardrail", "动作失败或风险升高时回退；最终经过规则护栏过滤。", "red"),
        t("→", { width: fixed(42), style: { fontSize: 35, color: colors.green, bold: true, alignment: "center" } }),
        stepBox("6", "执行与记录", "输出本步动作，记录 source、reward、violation 和诊断字段。"),
      ],
    ),
    panel({ width: fill, height: hug, padding: { x: 24, y: 16 }, fill: colors.white, line: { color: colors.line, weight: 1 }, borderRadius: 16 },
      t("关键理解：LLM 负责“方向和目标”，RSPC/guardrail 负责“每一步能不能安全执行”。", {
        style: { fontSize: 25, bold: true, color: colors.ink, alignment: "center" },
      })),
  ]),
  "这页回答项目整体怎么跑。用户之前的理解基本正确，但需要强调：现在我们把 LLM 输出拆成高层参考轨迹，底层 RSPC 才是实际每步闭环执行主体。",
);

addSlide(
  4,
  "当前控制架构：主控制器稳定，增强层默认关闭或诊断运行",
  "避免把尚未验证稳定的增强模块直接写入 baseline，所有模块都有明确边界。",
  grid(
    { width: fill, height: grow(1), columns: [fr(1.05), fr(1.1), fr(1.05)], rows: [auto, auto], columnGap: 26, rowGap: 22 },
    [
      moduleBox("PPO 诊断镜子", "用于观察 RL 在相同场景下的动作倾向；当前 PPO 不是最终强教师。", "blue", 176),
      moduleBox("LLM-RSPC 主控制器", "LLM 生成 reference plan；RSPC 进行加权 rollout、误差修正和 fallback。", "green", 176),
      moduleBox("HEM 经验库", "从 paired rollout 中挖掘湿度经济经验；当前收益叙事暂停。", "amber", 176),
      moduleBox("Plan Cache", "record/replay/refresh 三种模式冻结 LLM 计划，减少 API 随机性污染实验。", "blue", 176),
      moduleBox("MC-SERO Shadow", "后台生成机制候选和风险经济评分，只记录 would-select，不接管动作。", "green", 176),
      moduleBox("Tomato Safety v2", "默认关闭的番茄机理护栏，针对低 RH、高 VPD、冷恢复做 opt-in 验证。", "red", 176),
    ],
  ),
  "这一页要讲清楚模块边界：LLM-RSPC 是主线；PPO 是诊断，不是最终教师；HEM 是后续反事实经验学习；MC-SERO 仍是 shadow；v2 是 opt-in 护栏。",
);

addSlide(
  5,
  "LLM-RSPC 细节：计划、候选、护栏三层把 LLM 输出变成可执行动作",
  "一条动作不是直接来自 LLM，而是经过合同、候选池和安全过滤逐层收紧。",
  grid(
    { width: fill, height: grow(1), columns: [fr(1), fr(1), fr(1)], columnGap: 26 },
    [
      panel({ width: fill, height: fill, padding: 26, fill: colors.white, line: { color: colors.line, weight: 1 }, borderRadius: 18 },
        column({ width: fill, height: hug, gap: 18 }, [
          chip("高层规划", "blue", fixed(120)),
          t("LLM 输出", { style: { fontSize: 31, bold: true, color: colors.ink } }),
          bulletList(["锚点动作 anchor action", "未来 11 步目标状态", "策略意图与重规划原因", "异常时 fallback 信息"], { fontSize: 20 }),
        ])),
      panel({ width: fill, height: fill, padding: 26, fill: colors.white, line: { color: colors.line, weight: 1 }, borderRadius: 18 },
        column({ width: fill, height: hug, gap: 18 }, [
          chip("合同机制", "amber", fixed(120)),
          t("Setpoint Contract", { style: { fontSize: 31, bold: true, color: colors.ink } }),
          bulletList(["裁剪过激目标", "补齐缺失目标状态", "限制不可执行 CO2/RH/温度设定", "生成可回放 reference plan"], { fontSize: 20 }),
        ])),
      panel({ width: fill, height: fill, padding: 26, fill: colors.white, line: { color: colors.line, weight: 1 }, borderRadius: 18 },
        column({ width: fill, height: hug, gap: 18 }, [
          chip("闭环执行", "green", fixed(120)),
          t("Safe Rollout", { style: { fontSize: 31, bold: true, color: colors.ink } }),
          bulletList(["anchor / rule / blend 竞争", "误差修正跟踪目标", "fallback_plan 兜底", "guardrail 输出最终动作"], { fontSize: 20 }),
        ])),
    ],
  ),
  "这一页讲实现细节：LLM 不是最终执行者。即使 LLM 给出不完整计划，setpoint contract 会补齐；rollout 和 fallback 再决定本步控制；最后 guardrail 兜底。",
);

addSlide(
  6,
  "可复现评估体系：先冻结高层计划，再比较底层控制改动",
  "这是当前项目最关键的实验基础，避免 LLM 每次输出不同导致对比不公平。",
  column({ width: fill, height: grow(1), gap: 24 }, [
    grid(
      { width: fill, height: fixed(220), columns: [fr(1), fr(1), fr(1)], columnGap: 24 },
      [
        moduleBox("record", "调用 LLM 并写入 plan cache，保存 raw response、parsed plan、contract 后计划和 fallback 信息。", "green", 190),
        moduleBox("replay --strict", "不再调用新 LLM；cache miss 直接失败，确保同一高层计划下比较控制器。", "blue", 190),
        moduleBox("refresh", "明确需要更新计划时才覆盖旧缓存，用于新模型或新 prompt 的重新评估。", "amber", 190),
      ],
    ),
    row({ width: fill, height: fixed(270), gap: 28 }, [
      panel({ width: fixed(560), height: fill, padding: 28, fill: colors.white, line: { color: colors.line, weight: 1 }, borderRadius: 18 },
        column({ width: fill, height: hug, gap: 18 }, [
          t("Frozen Benchmark 检查项", { style: { fontSize: 28, bold: true, color: colors.ink } }),
          bulletList(["cache hit = steps", "record/replay 指标一致", "llm_sero_shadow 与 llm 行为完全一致", "unknown source 立即报错"], { fontSize: 20 }),
        ])),
      panel({ width: fill, height: fill, padding: 28, fill: "#EDF7F0", borderRadius: 18 },
        column({ width: fill, height: hug, gap: 18 }, [
          t("为什么这一步重要？", { style: { fontSize: 30, bold: true, color: colors.ink } }),
          t("后续 RSPC、HEM、MC-SERO、Tomato Safety v2 的差异，必须来自控制逻辑本身，而不是来自 LLM API 输出波动。", {
            style: { fontSize: 27, color: colors.ink, lineSpacing: 1.22 },
          }),
        ])),
    ]),
  ]),
  "这页要让老师放心：现在我们没有直接扩大实验，而是先做可复现评估底座。以后所有模块对比都可以在同一批 cached plans 下公平比较。",
);

addSlide(
  7,
  "HEM 阶段结论：经验库框架完成，但当前不作为收益主线",
  "湿度经济经验库解决“如何记住好经验”的问题，但当前闭环收益证据不足。",
  grid(
    { width: fill, height: grow(1), columns: [fr(1.3), fr(1)], columnGap: 32 },
    [
      column({ width: fill, height: hug, gap: 18 }, [
        row({ width: fill, height: fixed(118), gap: 14 }, [
          stepBox("1", "mine", "PPO/LLM 配对轨迹中筛选局部反事实经验。", "blue"),
          stepBox("2", "shadow", "离线判断经验命中与代理收益。", "green"),
        ]),
        row({ width: fill, height: fixed(118), gap: 14 }, [
          stepBox("3", "horizon filter", "拒绝把 RH 风险或补热成本推迟到未来。", "amber"),
          stepBox("4", "candidate", "仅在版本 metadata 匹配时进入候选池。", "red"),
        ]),
        panel({ width: fill, height: hug, padding: { x: 24, y: 18 }, fill: colors.white, line: { color: colors.line, weight: 1 }, borderRadius: 16 },
          t("当前判断：HEM 机制可以保留为后续 Counterfactual Experience Learning，但现阶段不宣称闭环性能收益。", {
            style: { fontSize: 24, bold: true, color: colors.ink, lineSpacing: 1.2 },
          })),
      ]),
      column({ width: fill, height: hug, gap: 24 }, [
        metric("已具备", "版本门控", "teacher_policy_id / baseline_controller_id / schema version", "green"),
        metric("当前问题", "selected=0", "pilot dryfix 下 HEM available 但未被闭环选中", "amber"),
        metric("下一定位", "反事实经验", "等强 PPO/SAC 与 RSPC 稳定后再重挖经验", "blue"),
      ]),
    ],
  ),
  "这一页要避免夸大：我们确实实现了经验库生命周期和安全过滤，但当前 HEM 没有稳定 selected 和收益证据，所以暂时不作为论文主要性能来源。",
);

addSlide(
  8,
  "MC-SERO Shadow：后台机制候选评价器，先诊断再接管",
  "它不改变动作，只回答：如果由机制评分器选择，它会倾向什么候选？",
  grid(
    { width: fill, height: grow(1), columns: [fr(1), fr(1.1)], columnGap: 32 },
    [
      column({ width: fill, height: hug, gap: 22 }, [
        moduleBox("候选集合", "economy_hold、economic_dehumidify、safe_dehumidify、emergency_dehumidify、dry_recovery、heat_buffer，加上 anchor/rule/blend。", "green", 188),
        moduleBox("评分窗口", "6-step deterministic proxy horizon，考虑温度、RH、VPD、结露、能耗和执行器冲突。", "blue", 188),
        moduleBox("当前状态", "shadow mode 不覆盖 selected_control；llm_sero_shadow 与 llm 指标必须完全一致。", "amber", 188),
      ]),
      panel({ width: fill, height: fill, padding: 28, fill: colors.white, line: { color: colors.line, weight: 1 }, borderRadius: 18 },
        column({ width: fill, height: hug, gap: 22 }, [
          t("Pilot 中 MC-SERO 产生非零诊断信号", { style: { fontSize: 30, bold: true, color: colors.ink } }),
          row({ width: fill, height: hug, gap: 24 }, [
            metric("2020/d240/s42", "105/240", "would-select steps", "green"),
            metric("2015/d120/s43", "102/240", "would-select steps", "blue"),
          ]),
          t("意义：它能定位 RSPC 可能需要改进的状态窗口，但还没有进入真实闭环选择。", {
            style: { fontSize: 24, color: colors.ink, lineSpacing: 1.2 },
          }),
        ])),
    ],
  ),
  "这页说明 MC-SERO 的价值：它现在是诊断层，不会污染 baseline。我们利用它观察机制候选是否合理，为之后 guarded select 做准备。",
);

addSlide(
  9,
  "Tomato Safety v2：番茄机理护栏，专门处理低 RH / 高 VPD / 冷恢复",
  "v2 默认关闭，只通过 llm_rspc_v2 标签 opt-in 验证，避免污染普通 LLM-RSPC baseline。",
  grid(
    { width: fill, height: grow(1), columns: [fr(1), fr(1), fr(1)], columnGap: 24 },
    [
      panel({ width: fill, height: fill, padding: 26, fill: colors.white, line: { color: colors.line, weight: 1 }, borderRadius: 18 },
        column({ width: fill, height: hug, gap: 18 }, [
          chip("dry guard", "amber", fixed(128)),
          t("干侧保护", { style: { fontSize: 31, bold: true, color: colors.ink } }),
          bulletList(["RH < 55 或 VPD > 1.2 触发", "限制通风、CO2、补光", "RH < 50 或 VPD > 1.6 强保护"], { fontSize: 20 }),
        ])),
      panel({ width: fill, height: fill, padding: 26, fill: colors.white, line: { color: colors.line, weight: 1 }, borderRadius: 18 },
        column({ width: fill, height: hug, gap: 18 }, [
          chip("cold recovery", "blue", fixed(154)),
          t("低温恢复", { style: { fontSize: 31, bold: true, color: colors.ink } }),
          bulletList(["temp_air < 15.5 且非极端结露", "禁止大通风", "优先保温与轻补热"], { fontSize: 20 }),
        ])),
      panel({ width: fill, height: fill, padding: 26, fill: colors.white, line: { color: colors.line, weight: 1 }, borderRadius: 18 },
        column({ width: fill, height: hug, gap: 18 }, [
          chip("replay safe", "green", fixed(142)),
          t("评估约束", { style: { fontSize: 31, bold: true, color: colors.ink } }),
          bulletList(["strict replay 下 cache hit 必须满步", "记录 suppressed replan", "不改变 llm baseline"], { fontSize: 20 }),
        ])),
    ],
  ),
  "这一页讲番茄机理：番茄控制不是只防高湿，低 RH 和高 VPD 也会导致作物胁迫。v2 是针对这个问题的保守护栏。",
);

addSlide(
  10,
  "Pilot 结果：v2 在两个代表场景上显著改善低湿和 VPD 风险",
  "注意：这是 pilot 级别的积极信号，还不能直接扩展为最终论文性能结论。",
  column({ width: fill, height: grow(1), gap: 22 }, [
    simpleTable(
      ["场景", "指标", "LLM", "LLM-RSPC v2", "变化", "解读"],
      [
        ["2015/d120/s43", "RH-low", "170.71", "4.59", "-97.3%", "低湿风险显著下降"],
        ["2015/d120/s43", "VPD-high", "21.35", "8.48", "-60.3%", "高 VPD 明显缓解"],
        ["2015/d120/s43", "Temp violation", "7.19", "5.47", "-23.9%", "温度安全未变差"],
        ["2020/d240/s42", "VPD-high", "0.67", "0.42", "-37.8%", "安全场景没有被破坏"],
      ],
      [fixed(230), fixed(210), fixed(170), fixed(220), fixed(150), fill],
    ),
    row({ width: fill, height: hug, gap: 28 }, [
      metric("2015 冷春场景", "v2 steps 61", "低 RH / 高 VPD / 冷恢复护栏有效触发", "green"),
      metric("2020 安全场景", "v2 steps 44", "reward/profit 未明显恶化，VPD 降低", "blue"),
      metric("PPO 角色", "诊断镜子", "当前 PPO 不是最终强教师，需要后续重训", "amber"),
    ]),
  ]),
  "讲这页时要谨慎：pilot 结果很好，说明方向有价值，但我们没有说 v2 已经稳定泛化；下一页会说明 sentinel gate 发现了跨季节风险。",
);

addSlide(
  11,
  "Sentinel Gate 结果：跨季节检查发现 v2 的高温强辐射副作用",
  "这不是无效失败，而是评估体系成功拦截 pilot 上看不出的泛化风险。",
  grid(
    { width: fill, height: grow(1), columns: [fr(1), fr(1)], columnGap: 34 },
    [
      panel({ width: fill, height: fill, padding: 30, fill: "#FFF8EC", line: { color: "#E8D4A7", weight: 1 }, borderRadius: 18 },
        column({ width: fill, height: hug, gap: 24 }, [
          chip("Gate stopped", "amber", fixed(150)),
          t("2010/day180/seed44", { style: { fontSize: 39, bold: true, color: colors.ink } }),
          metric("提前终止", "约 55 步", "未继续运行 2015/2020 shard", "amber"),
          metric("最高温度", "约 35.6°C", "高温强辐射下温度安全失败", "red"),
        ])),
      column({ width: fill, height: hug, gap: 22 }, [
        moduleBox("暴露的问题", "v2 的 dry/VPD guard 在夏季强辐射下过度偏向保湿，screen=1.0、shade=0.0 可能导致积热。", "red", 188),
        moduleBox("为什么停止", "sentinel gate 规则要求一旦出现 cache/source/安全失败就停止，避免把错误扩散到正式 36x240 证据。", "amber", 188),
        moduleBox("论文价值", "证明方法开发不是只在 pilot 上调参，而是通过跨季节 gate 检验机理护栏的泛化性。", "green", 188),
      ]),
    ],
  ),
  "这一页要坦诚讲：v2 在 pilot 上改善明显，但 sentinel 发现高温强辐射副作用。这个结果反而说明 frozen benchmark 和 gate 是必要的。",
);

addSlide(
  12,
  "下一阶段：先修 hot override，再恢复 sentinel gate 和强基线训练",
  "下一步重点不是堆模块，而是把已发现的泛化风险修掉，再扩大实验。",
  column({ width: fill, height: grow(1), gap: 26 }, [
    grid(
      { width: fill, height: hug, columns: [fr(1), auto, fr(1), auto, fr(1)], columnGap: 14, alignItems: "center" },
      [
        stepBox("1", "Hot Override 修复", "temp_air ≥ 32°C 或高辐射升温时，温度安全优先于保湿。", "red"),
        t("→", { width: fixed(42), style: { fontSize: 35, color: colors.green, bold: true, alignment: "center" } }),
        stepBox("2", "2010/d180 Canary", "验证 screen/shade/vent 不再造成积热提前终止。", "amber"),
        t("→", { width: fixed(42), style: { fontSize: 35, color: colors.green, bold: true, alignment: "center" } }),
        stepBox("3", "12x240 Sentinel", "seed=44 跨年份跨季节 gate 重新运行。", "green"),
      ],
    ),
    grid(
      { width: fill, height: hug, columns: [fr(1), auto, fr(1), auto, fr(1)], columnGap: 14, alignItems: "center" },
      [
        stepBox("4", "36x240 Frozen Benchmark", "通过 sentinel 后再进入正式多 seed benchmark。", "blue"),
        t("→", { width: fixed(42), style: { fontSize: 35, color: colors.green, bold: true, alignment: "center" } }),
        stepBox("5", "强 PPO/SAC Baseline", "固定 reward 版本，做超参搜索与 Pareto 选模。", "amber"),
        t("→", { width: fixed(42), style: { fontSize: 35, color: colors.green, bold: true, alignment: "center" } }),
        stepBox("6", "论文实验", "再跑 960 步、全周期与消融实验。", "green"),
      ],
    ),
    panel({ width: fill, height: hug, padding: { x: 28, y: 20 }, fill: colors.white, line: { color: colors.line, weight: 1 }, borderRadius: 18 },
      t("阶段结论：项目已经从“能跑”进入“可复现、可诊断、可解释地改进”的阶段；下一步是提升 v2 泛化安全，再建立强 RL 对照。", {
        style: { fontSize: 27, bold: true, color: colors.ink, alignment: "center", lineSpacing: 1.18 },
      })),
  ]),
  "最后总结：下一阶段不是继续堆 HEM 或扩大实验，而是修 hot override，通过 canary 和 12x240 sentinel 后再扩大到 36x240，并同步准备强 PPO/SAC baseline。",
);

await fs.mkdir(OUT.previewDir, { recursive: true });
const pptx = await PresentationFile.exportPptx(presentation);
await pptx.save(OUT.workspacePptx);
await fs.copyFile(OUT.workspacePptx, OUT.finalPptx);

for (let i = 0; i < presentation.slides.count; i += 1) {
  const slide = presentation.slides.getItem(i);
  const png = await slide.export({ format: "png" });
  const buffer = Buffer.from(await png.arrayBuffer());
  await fs.writeFile(path.join(OUT.previewDir, `slide_${String(i + 1).padStart(2, "0")}.png`), buffer);
}

console.log(JSON.stringify({
  slides: presentation.slides.count,
  pptx: path.resolve(OUT.finalPptx),
  workspace_pptx: path.resolve(OUT.workspacePptx),
  previews: path.resolve(OUT.previewDir),
}, null, 2));
