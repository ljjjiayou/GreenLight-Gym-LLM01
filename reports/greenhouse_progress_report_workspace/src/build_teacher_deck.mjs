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
} from "@oai/artifact-tool";

const OUT = {
  workspacePptx: "output/teacher_output.pptx",
  finalPptx: "../greenhouse_progress_report_teacher_20260501.pptx",
  previewDir: "scratch/teacher_previews",
};

const W = 1920;
const H = 1080;
const C = {
  bg: "#F6FAF4",
  pale: "#EAF5EC",
  ink: "#17352E",
  muted: "#607770",
  green: "#2E7D5B",
  greenSoft: "#DDEFE5",
  blue: "#2F6F9F",
  blueSoft: "#D9EAF5",
  amber: "#B7791F",
  amberSoft: "#F4E2BD",
  red: "#B4443F",
  redSoft: "#F2D8D5",
  white: "#FFFFFF",
  line: "#CADBD1",
  dark: "#0F2B24",
};

const presentation = Presentation.create({ slideSize: { width: W, height: H } });

function txt(value, opts = {}) {
  return text(value, {
    name: opts.name,
    width: opts.width ?? fill,
    height: opts.height ?? hug,
    columnSpan: opts.columnSpan,
    rowSpan: opts.rowSpan,
    style: {
      fontSize: 24,
      color: C.ink,
      lineSpacing: 1.22,
      ...(opts.style || {}),
    },
  });
}

function chip(label, tone = "green", width = hug) {
  const fillColor = tone === "blue" ? C.blueSoft : tone === "amber" ? C.amberSoft : tone === "red" ? C.redSoft : C.greenSoft;
  const color = tone === "blue" ? C.blue : tone === "amber" ? C.amber : tone === "red" ? C.red : C.green;
  return panel(
    { width, height: hug, padding: { x: 18, y: 8 }, fill: fillColor, borderRadius: 18 },
    txt(label, { style: { fontSize: 17, bold: true, color, alignment: "center" } }),
  );
}

function bullet(items, opts = {}) {
  return column(
    { width: opts.width ?? fill, height: hug, gap: opts.gap ?? 12 },
    items.map((item) =>
      row({ width: fill, height: hug, gap: 12, align: "start" }, [
        txt("•", { width: fixed(20), style: { fontSize: opts.fontSize ?? 22, bold: true, color: opts.color ?? C.green } }),
        txt(item, { width: fill, style: { fontSize: opts.fontSize ?? 21, color: opts.textColor ?? C.ink, lineSpacing: 1.2 } }),
      ]),
    ),
  );
}

function metric(label, value, note, tone = "green") {
  const color = tone === "blue" ? C.blue : tone === "amber" ? C.amber : tone === "red" ? C.red : C.green;
  return column({ width: fill, height: hug, gap: 7 }, [
    txt(label, { style: { fontSize: 17, bold: true, color: C.muted } }),
    txt(value, { style: { fontSize: 42, bold: true, color, lineSpacing: 1.0 } }),
    txt(note, { style: { fontSize: 16, color: C.muted, lineSpacing: 1.13 } }),
  ]);
}

function card(title, body, tone = "green", h = 160) {
  const fillColor = tone === "blue" ? "#EFF7FC" : tone === "amber" ? "#FFF7E8" : tone === "red" ? "#FFF0F0" : C.white;
  const color = tone === "blue" ? C.blue : tone === "amber" ? C.amber : tone === "red" ? C.red : C.green;
  return panel(
    {
      width: fill,
      height: fixed(h),
      padding: { x: 24, y: 18 },
      fill: fillColor,
      line: { color: C.line, weight: 1 },
      borderRadius: 16,
    },
    column({ width: fill, height: hug, gap: 10 }, [
      row({ width: fill, height: hug, gap: 12, align: "center" }, [
        shape({ width: fixed(10), height: fixed(28), fill: color, borderRadius: 5 }),
        txt(title, { style: { fontSize: 23, bold: true, color: C.ink } }),
      ]),
      txt(body, { style: { fontSize: 18, color: C.muted, lineSpacing: 1.18 } }),
    ]),
  );
}

function moduleExplain(title, problem, method, benefit, tone = "green") {
  const color = tone === "blue" ? C.blue : tone === "amber" ? C.amber : tone === "red" ? C.red : C.green;
  return panel(
    { width: fill, height: fill, padding: 24, fill: C.white, line: { color: C.line, weight: 1 }, borderRadius: 16 },
    column({ width: fill, height: hug, gap: 16 }, [
      chip(title, tone, fixed(210)),
      row({ width: fill, height: hug, gap: 10 }, [
        txt("解决什么", { width: fixed(86), style: { fontSize: 17, bold: true, color } }),
        txt(problem, { style: { fontSize: 18, color: C.ink } }),
      ]),
      row({ width: fill, height: hug, gap: 10 }, [
        txt("怎么做", { width: fixed(86), style: { fontSize: 17, bold: true, color } }),
        txt(method, { style: { fontSize: 18, color: C.ink } }),
      ]),
      row({ width: fill, height: hug, gap: 10 }, [
        txt("收益", { width: fixed(86), style: { fontSize: 17, bold: true, color } }),
        txt(benefit, { style: { fontSize: 18, color: C.ink } }),
      ]),
    ]),
  );
}

function header(no, title, subtitle) {
  return column({ width: fill, height: hug, gap: 10 }, [
    row({ width: fill, height: hug, align: "center" }, [
      txt(String(no).padStart(2, "0"), { width: fixed(58), style: { fontSize: 19, bold: true, color: C.green } }),
      rule({ width: fixed(90), stroke: C.green, weight: 3 }),
      txt("温室控制项目进度汇报｜导师友好版", { width: fill, style: { fontSize: 17, color: C.muted, alignment: "right" } }),
    ]),
    txt(title, { style: { fontSize: 46, bold: true, color: C.ink, lineSpacing: 1.08 } }),
    subtitle ? txt(subtitle, { width: wrap(1420), style: { fontSize: 24, color: C.muted, lineSpacing: 1.22 } }) : null,
  ].filter(Boolean));
}

function addSlide(no, title, subtitle, body, notes) {
  const slide = presentation.slides.add();
  slide.compose(
    layers({ width: fill, height: fill }, [
      shape({ width: fill, height: fill, fill: C.bg }),
      shape({ width: fixed(470), height: fill, fill: no % 2 ? "#EDF7F0" : "#F1F8F1" }),
      column({ width: fill, height: fill, padding: { x: 84, y: 58 }, gap: 26 }, [
        header(no, title, subtitle),
        body,
        row({ width: fill, height: hug, align: "center" }, [
          txt("数据来源：项目文档、pilot frozen replay 与 sentinel gate 诊断输出", { width: fill, style: { fontSize: 13, color: "#7C918A" } }),
          txt(`${no}/12`, { width: fixed(74), style: { fontSize: 13, color: "#7C918A", alignment: "right" } }),
        ]),
      ]),
    ]),
    { frame: { left: 0, top: 0, width: W, height: H }, baseUnit: 8 },
  );
  slide.speakerNotes.setText(notes);
}

function step(num, title, body, tone = "green") {
  const fillColor = tone === "blue" ? C.blueSoft : tone === "amber" ? C.amberSoft : tone === "red" ? C.redSoft : C.greenSoft;
  const color = tone === "blue" ? C.blue : tone === "amber" ? C.amber : tone === "red" ? C.red : C.green;
  return panel(
    { width: fill, height: fixed(132), padding: { x: 22, y: 17 }, fill: fillColor, borderRadius: 18 },
    row({ width: fill, height: fill, gap: 14, align: "start" }, [
      panel({ width: fixed(40), height: fixed(40), fill: C.white, borderRadius: 20 },
        txt(num, { style: { fontSize: 18, bold: true, color, alignment: "center" } })),
      column({ width: fill, height: hug, gap: 8 }, [
        txt(title, { style: { fontSize: 22, bold: true, color: C.ink } }),
        txt(body, { style: { fontSize: 17, color: C.muted, lineSpacing: 1.15 } }),
      ]),
    ]),
  );
}

function resultTable() {
  const headers = ["场景", "指标", "原 LLM-RSPC", "v2 后", "收益"];
  const rows = [
    ["2015/d120/s43", "RH-low", "170.71", "4.59", "低湿风险下降 97.3%"],
    ["2015/d120/s43", "VPD-high", "21.35", "8.48", "高 VPD 下降 60.3%"],
    ["2015/d120/s43", "温度违规", "7.19", "5.47", "温度安全未变差"],
    ["2020/d240/s42", "VPD-high", "0.67", "0.42", "安全场景仍稳定"],
  ];
  const widths = [fixed(250), fixed(210), fixed(220), fixed(170), fill];
  return column({ width: fill, height: hug, gap: 0 }, [
    row({ width: fill, height: fixed(54), gap: 0 }, headers.map((h, i) =>
      panel({ width: widths[i], height: fill, fill: C.green, padding: { x: 14, y: 12 } },
        txt(h, { style: { fontSize: 17, bold: true, color: C.white, alignment: i < 2 ? "left" : "center" } })),
    )),
    ...rows.map((r, idx) =>
      row({ width: fill, height: fixed(64), gap: 0 }, r.map((v, i) =>
        panel({ width: widths[i], height: fill, fill: idx % 2 ? "#F0F7F2" : C.white, line: { color: "#E1EAE4", weight: 1 }, padding: { x: 14, y: 12 } },
          txt(v, { style: { fontSize: i === 4 ? 18 : 17, bold: i === 4, color: i === 4 ? C.green : C.ink, alignment: i < 2 ? "left" : "center" } })),
      )),
    ),
  ]);
}

function cover() {
  const slide = presentation.slides.add();
  slide.compose(
    layers({ width: fill, height: fill }, [
      shape({ width: fill, height: fill, fill: C.dark }),
      shape({ width: fixed(560), height: fill, fill: "#173D32" }),
      column({ width: fill, height: fill, padding: { x: 100, y: 76 }, gap: 34 }, [
        row({ width: fill, height: hug, gap: 12 }, [
          chip("整体流程", "green", fixed(130)),
          chip("模块作用", "blue", fixed(130)),
          chip("测试收益", "amber", fixed(130)),
        ]),
        column({ width: wrap(1220), height: hug, gap: 22 }, [
          txt("温室控制项目进度汇报", { style: { fontSize: 76, bold: true, color: C.white, lineSpacing: 1.02 } }),
          txt("导师友好版：讲清楚系统怎么工作、每个模块为什么存在、目前测试带来了什么收益", {
            width: wrap(1180),
            style: { fontSize: 32, color: "#CFE6D8", lineSpacing: 1.14 },
          }),
        ]),
        row({ width: fill, height: fixed(250), gap: 34, align: "end" }, [
          metric("主控制框架", "LLM-RSPC", "LLM 负责高层目标，RSPC 负责安全执行", "green"),
          metric("评估方法", "Frozen Replay", "冻结 LLM 输出，公平比较模块改动", "blue"),
          metric("当前阶段", "发现并修正风险", "pilot 有收益，sentinel 暴露泛化问题", "amber"),
        ]),
        txt("2026-05-01", { style: { fontSize: 20, color: "#A9C7BA" } }),
      ]),
    ]),
    { frame: { left: 0, top: 0, width: W, height: H }, baseUnit: 8 },
  );
  slide.speakerNotes.setText("这一版汇报的目标是降低理解门槛。先告诉老师：项目不是单个算法，而是一个温室闭环控制系统。后面每个模块都按“解决什么、怎么做、收益是什么”来讲。");
}

cover();

addSlide(
  2,
  "先用一句话讲清楚：项目要解决什么问题",
  "温室控制不是只追求 reward，而是在作物安全、能耗成本和执行器约束之间做稳定权衡。",
  grid({ width: fill, height: grow(1), columns: [fr(1), fr(1)], rows: [auto, auto], columnGap: 28, rowGap: 22 }, [
    card("控制对象", "模拟番茄温室，状态包括温度、湿度、VPD、CO2、辐射、天气和作物状态。", "green", 172),
    card("控制动作", "加热、CO2、保温幕、通风、补光、遮阳等执行器组合。", "blue", 172),
    card("核心矛盾", "除湿可能要通风，但通风会带走热量和 CO2；保湿可能降低 VPD，但也可能造成积热。", "amber", 172),
    card("项目目标", "让 LLM 给出高层计划，同时用规则、预测和护栏保证每一步动作安全可执行。", "green", 172),
  ]),
  "这一页不讲模块名，先讲控制问题本身：温室控制有多目标冲突，特别是湿度、温度、能耗之间互相牵制，所以需要分层控制和安全评估。",
);

addSlide(
  3,
  "整体闭环流程：从当前温室状态到最终执行动作",
  "可以把系统理解成“先想清楚目标，再安全地执行一步，然后继续观察修正”。",
  column({ width: fill, height: grow(1), gap: 26 }, [
    grid({ width: fill, height: hug, columns: [fr(1), auto, fr(1), auto, fr(1)], columnGap: 14, alignItems: "center" }, [
      step("1", "读取温室状态", "获得 T/RH/VPD/CO2/天气/作物状态，判断当前风险。"),
      txt("→", { width: fixed(42), style: { fontSize: 34, bold: true, color: C.green, alignment: "center" } }),
      step("2", "LLM 生成高层计划", "给出锚点动作和未来目标状态，不直接裸执行。", "blue"),
      txt("→", { width: fixed(42), style: { fontSize: 34, bold: true, color: C.green, alignment: "center" } }),
      step("3", "计划合同修正", "把 LLM 目标补齐、裁剪成可执行 reference plan。", "amber"),
    ]),
    grid({ width: fill, height: hug, columns: [fr(1), auto, fr(1), auto, fr(1)], columnGap: 14, alignItems: "center" }, [
      step("4", "RSPC 滚动控制", "每一步比较当前状态和目标状态，生成候选动作。", "blue"),
      txt("→", { width: fixed(42), style: { fontSize: 34, bold: true, color: C.green, alignment: "center" } }),
      step("5", "Fallback 与护栏", "动作不可靠时回退，最终检查温湿度与执行器冲突。", "red"),
      txt("→", { width: fixed(42), style: { fontSize: 34, bold: true, color: C.green, alignment: "center" } }),
      step("6", "执行并记录", "执行本步动作，记录 reward、成本、违规和模块诊断。"),
    ]),
    panel({ width: fill, height: hug, padding: { x: 26, y: 16 }, fill: C.white, line: { color: C.line, weight: 1 }, borderRadius: 16 },
      txt("给导师的解释口径：LLM 像“高层种植顾问”，RSPC 和护栏像“现场安全执行员”。", {
        style: { fontSize: 26, bold: true, alignment: "center", color: C.ink },
      })),
  ]),
  "这一页是核心流程图。重点说：每一步都不是一次性规划完就不管，而是滚动控制。LLM 给方向，底层每一步根据真实状态修正。",
);

addSlide(
  4,
  "主模块 1：LLM 高层规划负责“想目标”，不是直接控制设备",
  "导师如果不熟悉 LLM，可以把它理解为根据当前状态给出短期管理建议。",
  grid({ width: fill, height: grow(1), columns: [fr(1), fr(1.05)], columnGap: 30 }, [
    moduleExplain(
      "LLM Planner",
      "单纯规则难以在复杂天气和多目标冲突下给出高层策略。",
      "读取温室状态摘要，输出锚点动作、未来 11 步目标状态和重规划理由。",
      "让控制器有“接下来往哪里走”的参考，而不是只看当前一步。",
      "blue",
    ),
    column({ width: fill, height: hug, gap: 20 }, [
      card("输入", "当前温室状态、天气、作物阶段、历史计划执行情况。", "green", 150),
      card("输出", "anchor action、target temperature/RH/CO2 profile、策略意图、fallback 信息。", "blue", 150),
      card("保护机制", "LLM 输出不会直接执行，必须经过合同修正、候选竞争和安全护栏。", "amber", 150),
    ]),
  ]),
  "这一页要降低老师对 LLM 的误解：不是让大模型直接控制加热和通风，而是让它提供高层参考计划。",
);

addSlide(
  5,
  "主模块 2：Setpoint Contract 把 LLM 计划变成可执行目标",
  "它的作用类似“审稿人”：检查 LLM 目标是否缺失、过激或不可执行。",
  grid({ width: fill, height: grow(1), columns: [fr(1), fr(1), fr(1)], columnGap: 24 }, [
    moduleExplain("补齐", "LLM 有时不给完整未来目标。", "用合同机制填充未来温室目标状态。", "保证 rollout 有连续参考轨迹。", "green"),
    moduleExplain("裁剪", "LLM 目标可能过低湿、过高温或不可执行。", "把温度、RH、CO2 目标限制在安全范围。", "减少异常 LLM 输出带来的风险。", "amber"),
    moduleExplain("记录", "实验需要知道目标是 LLM 原始给的还是合同修正后的。", "保存 raw response、parsed plan 和 contract 后计划。", "方便论文复现和失败分析。", "blue"),
  ]),
  "这一页强调计划合同的重要性。它不是算法花活，而是把大模型输出转成可控工程对象的关键。",
);

addSlide(
  6,
  "主模块 3：RSPC Rollout 和 Guardrail 负责“每一步怎么安全执行”",
  "底层控制不是照抄计划，而是每一步都根据误差、候选和安全规则重新选择动作。",
  grid({ width: fill, height: grow(1), columns: [fr(1.2), fr(1)], columnGap: 32 }, [
    column({ width: fill, height: hug, gap: 18 }, [
      card("RSPC Rollout", "读取当前状态与下一目标状态，结合 anchor/rule/blend 候选，计算本步控制动作。", "blue", 152),
      card("Fallback Plan", "如果 LLM 计划缺失或动作生成失败，使用规则候选或保守 fallback 继续控制。", "amber", 152),
      card("Safety Guardrail", "最终检查温度、湿度、VPD、结露和执行器冲突，必要时覆盖动作。", "red", 152),
    ]),
    panel({ width: fill, height: fill, padding: 30, fill: C.white, line: { color: C.line, weight: 1 }, borderRadius: 18 },
      column({ width: fill, height: hug, gap: 20 }, [
        txt("这个模块带来的收益", { style: { fontSize: 31, bold: true, color: C.ink } }),
        bullet([
          "LLM 出错时系统仍能继续运行",
          "动作有来源记录，能解释为什么这样控制",
          "可以插入 HEM、MC-SERO、v2 等增强层做消融",
          "把“计划”变成“安全执行的一步动作”",
        ], { fontSize: 22 }),
      ])),
  ]),
  "这一页讲底层执行。要强调每一步动作都不是随意的，而是候选竞争和护栏过滤后的结果。",
);

addSlide(
  7,
  "辅助模块：PPO、HEM、MC-SERO 分别承担不同角色",
  "这三个模块容易让老师混淆，汇报时要明确：它们不是同一个层级的主控制器。",
  grid({ width: fill, height: grow(1), columns: [fr(1), fr(1), fr(1)], columnGap: 24 }, [
    moduleExplain(
      "PPO",
      "需要一个学习型控制器作为对照和诊断参考。",
      "在相同场景下观察 PPO 倾向于怎么加热、通风、保湿或除湿。",
      "目前只作为“诊断镜子”，后续需要训练更强 PPO/SAC baseline。",
      "blue",
    ),
    moduleExplain(
      "HEM 经验库",
      "希望把历史上好的局部控制经验保存下来。",
      "从 PPO 与 LLM-RSPC 配对轨迹中挖掘安全且更经济的湿度经验。",
      "框架已搭建，但当前 selected 不稳定，暂不作为收益主线。",
      "amber",
    ),
    moduleExplain(
      "MC-SERO Shadow",
      "需要知道当前 RSPC 在哪些状态下可能有更好机制候选。",
      "后台生成候选并评分，只记录 would-select，不改变实际动作。",
      "提供诊断信号，后续可发展成 guarded select。",
      "green",
    ),
  ]),
  "这一页是给导师澄清模块关系：PPO 是对照，HEM 是记忆库，MC-SERO 是后台诊断器；它们目前都不是替代 LLM-RSPC 的主控制器。",
);

addSlide(
  8,
  "评估模块：Frozen Replay 让实验对比公平、可复现",
  "如果每次都重新调用 LLM，结果变化可能来自 LLM 输出波动，而不是控制器改进。",
  grid({ width: fill, height: grow(1), columns: [fr(1), fr(1)], columnGap: 32 }, [
    column({ width: fill, height: hug, gap: 20 }, [
      card("record", "第一次运行时调用 LLM，并把高层计划写入 plan cache。", "green", 150),
      card("replay --strict", "后续实验只读取相同 cached plan；如果 miss 就直接报错。", "blue", 150),
      card("protocol check", "检查 cache hit、record/replay 一致性、shadow 是否污染 baseline。", "amber", 150),
    ]),
    panel({ width: fill, height: fill, padding: 30, fill: C.white, line: { color: C.line, weight: 1 }, borderRadius: 18 },
      column({ width: fill, height: hug, gap: 24 }, [
        txt("它给项目带来的收益", { style: { fontSize: 32, bold: true, color: C.ink } }),
        metric("公平性", "同一批计划", "比较的是底层控制逻辑，而不是 LLM 随机性", "green"),
        metric("可复现", "cache hit = steps", "结果可以被重新回放和检查", "blue"),
        metric("可诊断", "失败即停止", "sentinel gate 能及时拦截泛化风险", "amber"),
      ])),
  ]),
  "这一页强调评估体系的价值。它让我们现在发现问题不是坏事，而是因为系统已经具备拦截错误的能力。",
);

addSlide(
  9,
  "Tomato Safety v2：针对番茄生长机理补上的安全护栏",
  "当前最重要的机理问题是：不能只防高湿，也要防低湿和高 VPD。",
  grid({ width: fill, height: grow(1), columns: [fr(1), fr(1), fr(1)], columnGap: 24 }, [
    moduleExplain("低湿保护", "RH 太低会让作物失水压力增大。", "RH < 55 或 VPD > 1.2 时限制通风、CO2 和补光。", "降低低 RH 与高 VPD 违规。", "amber"),
    moduleExplain("低温恢复", "夜间或冷春场景下大通风会让温度恢复失败。", "低温且非极端结露时，优先保温和轻补热。", "减少冷恢复窗口中的温度风险。", "blue"),
    moduleExplain("默认关闭", "新护栏可能有泛化副作用。", "通过 llm_rspc_v2 opt-in 验证，不污染普通 LLM-RSPC baseline。", "方便做严格消融和回退。", "green"),
  ]),
  "这一页重点从番茄生长角度解释 v2。老师不需要知道代码细节，只要理解：这是针对番茄低湿、高 VPD、低温恢复的保护层。",
);

addSlide(
  10,
  "测试结果与阶段收益：pilot 上 v2 明显降低低湿和 VPD 风险",
  "这些结果说明方向有价值，但还需要通过跨季节 sentinel gate 才能作为正式性能结论。",
  column({ width: fill, height: grow(1), gap: 24 }, [
    resultTable(),
    row({ width: fill, height: hug, gap: 28 }, [
      metric("2015 冷春场景", "v2 steps 61", "低湿、高 VPD 和冷恢复护栏有效触发", "green"),
      metric("2020 安全场景", "v2 steps 44", "没有明显破坏原本安全场景", "blue"),
      metric("评估公平性", "240/240", "两个 pilot 场景 strict replay cache 全命中", "amber"),
    ]),
  ]),
  "这一页用数字讲收益。强调 RH-low 和 VPD-high 的下降，同时提醒老师：pilot 只是积极信号，不是最终大规模结论。",
);

addSlide(
  11,
  "同时也发现了问题：sentinel gate 拦截了一个跨季节泛化风险",
  "这页要坦诚讲：当前 v2 不是最终稳定版本，但我们已经知道下一步该修哪里。",
  grid({ width: fill, height: grow(1), columns: [fr(1), fr(1)], columnGap: 34 }, [
    panel({ width: fill, height: fill, padding: 30, fill: "#FFF8EC", line: { color: "#E7D3A3", weight: 1 }, borderRadius: 18 },
      column({ width: fill, height: hug, gap: 24 }, [
        chip("发现问题", "amber", fixed(130)),
        txt("2010/day180/seed44", { style: { fontSize: 38, bold: true, color: C.ink } }),
        metric("提前终止", "约 55 步", "停止后续 2015/2020 sentinel shard", "amber"),
        metric("最高温度", "约 35.6°C", "高温强辐射下温度安全失败", "red"),
      ])),
    column({ width: fill, height: hug, gap: 22 }, [
      card("原因判断", "v2 在高温强辐射下过度偏向保湿，screen=1.0、shade=0.0 可能导致积热。", "red", 168),
      card("为什么这是收益", "如果没有 frozen replay 与 sentinel gate，pilot 上的好结果可能会被误认为已经泛化。", "amber", 168),
      card("下一步方向", "加入 hot override：高温时优先降温，释放 screen、启用 shade、允许足够通风。", "green", 168),
    ]),
  ]),
  "这一页要把失败讲成诊断价值：系统没有盲目扩大实验，而是及时发现 v2 的高温副作用，下一步修 hot override。",
);

addSlide(
  12,
  "下一阶段工作计划：先修安全泛化，再做正式对比实验",
  "接下来不要继续堆模块，而是把当前发现的问题闭环修掉。",
  column({ width: fill, height: grow(1), gap: 24 }, [
    grid({ width: fill, height: hug, columns: [fr(1), auto, fr(1), auto, fr(1)], columnGap: 14, alignItems: "center" }, [
      step("1", "修 hot override", "高温/强辐射时温度安全优先于干侧保湿。", "red"),
      txt("→", { width: fixed(42), style: { fontSize: 34, bold: true, color: C.green, alignment: "center" } }),
      step("2", "2010/d180 canary", "先验证已知失败场景不再提前终止。", "amber"),
      txt("→", { width: fixed(42), style: { fontSize: 34, bold: true, color: C.green, alignment: "center" } }),
      step("3", "12x240 sentinel", "跨年份、跨季节、未调参 seed 检查泛化。", "green"),
    ]),
    grid({ width: fill, height: hug, columns: [fr(1), auto, fr(1), auto, fr(1)], columnGap: 14, alignItems: "center" }, [
      step("4", "36x240 benchmark", "通过 sentinel 后再做正式多 seed frozen benchmark。", "blue"),
      txt("→", { width: fixed(42), style: { fontSize: 34, bold: true, color: C.green, alignment: "center" } }),
      step("5", "强 PPO/SAC", "固定 reward 版本，训练更强 RL baseline。", "amber"),
      txt("→", { width: fixed(42), style: { fontSize: 34, bold: true, color: C.green, alignment: "center" } }),
      step("6", "论文实验", "再做 960 步、全周期和模块消融实验。", "green"),
    ]),
    panel({ width: fill, height: hug, padding: { x: 28, y: 18 }, fill: C.white, line: { color: C.line, weight: 1 }, borderRadius: 18 },
      txt("阶段总结：目前项目已经形成“可解释控制流程 + 可复现评估 + 可诊断改进”的体系，下一步重点是把泛化安全做扎实。", {
        style: { fontSize: 27, bold: true, alignment: "center", color: C.ink, lineSpacing: 1.16 },
      })),
  ]),
  "收尾时建议对老师说：目前不是没有进展，而是已经从能跑进入到可复现、可诊断、可迭代优化阶段。下一步的目标是 hot override 和强 baseline。",
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
