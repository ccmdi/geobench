import { AXES, METRICS, ROWS, SERIES, fmt, mean, pivot, sorted, writeState } from "./data.js";
import { fill, flag, h, select, swatch } from "./dom.js";

function tinted(metric, value) {
  return value == null ? null : `--g: ${Math.max(0, Math.min(1, metric.goodness(value))).toFixed(3)}`;
}

function guessed(answers) {
  return answers.map((a) => (a.refused ? "Refused" : `${a.guess_place?.name ?? "No country"}, ${fmt.km(a.distance_km)}`)).join("\n");
}

export function tableControls(state) {
  return [
    h("label", null, "Rows", select(state.rows, ROWS.map((key) => [key, AXES[key].name]), (rows) => writeState({ rows }))),
    h("label", null, "Columns", select(state.by, SERIES.map((key) => [key, AXES[key].name]), (by) => writeState({ by }))),
    h("label", null, "Measure", select(state.metric, Object.entries(METRICS).map(([key, m]) => [key, m.name]), (metric) => writeState({ metric }))),
    state.round ? h("label", null, "Rounds", select(state.scope, [["all", "All rounds"], ["round", "Active round"]], (scope) => writeState({ scope }))) : null,
  ];
}

export function tableView(ctx, view) {
  const { state } = ctx;
  const rows = AXES[state.rows];
  const metric = METRICS[state.metric];
  const scoped = state.scope === "round" && state.round ? ctx.answers.filter((a) => a.task === state.round) : ctx.answers;
  const { columns, lines } = pivot(scoped, rows, AXES[state.by], metric, ctx.labels);
  const header = (field, label, title) => {
    const active = state.sort.replace(/^-/, "") === field;
    const next = active && state.sort.startsWith("-") ? field : `-${field}`;
    return h("th", { class: active ? `sorted ${state.sort.startsWith("-") ? "desc" : "asc"}` : null, title, onclick: () => writeState({ sort: next }) }, label);
  };
  const opens = state.rows === "location";
  view.replaceChildren(
    h("div", { class: "table-wrap" },
      h("table", { class: "pivot" },
        h("thead", null, h("tr", null,
          header("label", rows.name),
          header("n", "n", "Rounds"),
          header("mean", "Mean"),
          header("spread", "Spread", "Highest minus lowest column"),
          columns.map((column, i) => header(`c${i}`, h("span", { class: "column" }, swatch(ctx.color(column.key)), column.label), column.label)),
        )),
        h("tbody", null, sorted(lines, state.sort).map((line) => {
          const bySeries = Map.groupBy(line.answers, AXES[state.by].key);
          return h("tr", { class: opens ? (line.key === state.round ? "opens current" : "opens") : null, onclick: opens ? () => writeState({ round: line.key }, { push: true }) : null },
            h("th", { scope: "row" }, rows.place ? flag(rows.place(line.answers[0])) : null, line.label, opens && new Set(ctx.answers.map((a) => a.map)).size > 1 ? h("span", { class: "muted" }, ` ${line.answers[0].map}`) : null),
            h("td", { class: "num muted" }, line.n),
            h("td", { class: "num cell", style: tinted(metric, line.mean) }, metric.format(line.mean)),
            h("td", { class: "num muted" }, line.spread == null ? "-" : metric.format(line.spread)),
            line.cells.map((value, i) =>
              h("td", { class: "num cell", style: tinted(metric, value), title: opens ? guessed(bySeries.get(columns[i].key) ?? []) : null }, metric.format(value)),
            ),
          );
        })),
      ),
    ),
  );
}

const SCORE_EDGES = Array.from({ length: 21 }, (_, i) => i * 250);
const KM_EDGES = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000];

function histogram(title, series, value, edges, label) {
  const [width, height, left, bottom, top] = [640, 240, 44, 28, 12];
  const bins = edges.length - 1;
  const shares = series.map((s) => {
    const values = s.answers.map(value).filter((v) => v != null);
    const counts = Array(bins).fill(0);
    for (const v of values) counts[Math.max(0, Math.min(bins - 1, edges.findLastIndex((edge) => edge <= v)))]++;
    return counts.map((count) => (values.length ? count / values.length : 0));
  });
  const peak = Math.max(0.01, ...shares.flat());
  const x = (i) => left + ((width - left - 8) * i) / bins;
  const y = (share) => top + (height - top - bottom) * (1 - share / peak);
  const tip = h("div", { class: "tip", hidden: true });
  const band = h("svg:rect", { class: "band", y: top, height: height - top - bottom, width: x(1) - x(0), visibility: "hidden" });
  const show = (i) => {
    band.setAttribute("x", x(i));
    band.setAttribute("visibility", "visible");
    tip.hidden = false;
    tip.style.left = `${(100 * (x(i) + x(i + 1))) / 2 / width}%`;
    fill(tip, h("strong", null, `${label(edges[i])} to ${label(edges[i + 1])}`), series.map((s, k) => h("div", null, swatch(s.color), `${s.label} `, h("span", { class: "num" }, fmt.pct(shares[k][i])))));
  };
  const hide = () => ((tip.hidden = true), band.setAttribute("visibility", "hidden"));
  const ticks = [0, peak / 2, peak];
  return h("figure", { class: "histogram" },
    h("figcaption", null, title),
    h("div", { class: "plot" },
      h("svg:svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": title, onmouseleave: hide },
        ticks.map((share) => [
          h("svg:line", { class: "grid", x1: left, x2: width - 8, y1: y(share), y2: y(share) }),
          h("svg:text", { class: "axis", x: left - 6, y: y(share) + 4, "text-anchor": "end" }, fmt.pct(share).replace(".0%", "%")),
        ]),
        edges.map((edge, i) => (i % Math.ceil(bins / 7) === 0 || i === bins ? h("svg:text", { class: "axis", x: x(i), y: height - 8, "text-anchor": "middle" }, label(edge)) : null)),
        band,
        shares.map((row, k) =>
          h("svg:path", { class: "step", stroke: series[k].color, d: row.map((share, i) => `${i ? "L" : "M"}${x(i)},${y(share)} H${x(i + 1)}`).join(" ") }),
        ),
        Array.from({ length: bins }, (_, i) => h("svg:rect", { class: "target", x: x(i), y: top, width: x(1) - x(0), height: height - top - bottom, onmouseenter: () => show(i) })),
      ),
      tip,
    ),
  );
}

export function scoresView(ctx, view) {
  const km = (v) => (v >= 1000 ? `${v / 1000}k` : `${v}`);
  view.replaceChildren(
    h("div", { class: "scores" },
      histogram("Score", ctx.series, (a) => a.points, SCORE_EDGES, (v) => (v >= 1000 ? `${v / 1000}k` : `${v}`)),
      histogram("Distance (km)", ctx.series, (a) => a.distance_km, KM_EDGES, km),
      h("table", { class: "summary" },
        h("thead", null, h("tr", null, h("th", null, AXES[ctx.state.by].name), h("th", null, "Answers"), h("th", null, "Mean score"), h("th", null, "Mean distance"), h("th", null, "Country"), h("th", null, "Refused"))),
        h("tbody", null, ctx.series.map((s) =>
          h("tr", null,
            h("th", { scope: "row" }, swatch(s.color), s.label),
            h("td", { class: "num" }, s.answers.length),
            [METRICS.points, METRICS.km, METRICS.hit].map((metric) => h("td", { class: "num" }, metric.format(mean(s.answers.map(metric.value))))),
            h("td", { class: "num" }, s.answers.filter((a) => a.refused).length),
          ),
        )),
      ),
    ),
  );
}

const SETTINGS = [
  ["Model", (s) => s.model],
  ["Lab", (s) => s.lab_name],
  ["Provider", (s, card) => card.provider ?? "-"],
  ["Map", (s) => s.map],
  ["Shelf", (s) => s.status],
  ["Started", (s, card) => card.started.slice(0, 16).replace("T", " ")],
  ["Temperature", (s) => s.temperature ?? "default"],
  ["Effort", (s) => s.effort ?? "-"],
  ["Thinking", (s) => (s.thinking ? "yes" : "no")],
  ["Search", (s) => (s.search ? "yes" : "no")],
  ["Token limit", (s) => fmt.int(s.max_tokens)],
  ["Settings", (s, card) => Object.entries(card.settings).map(([key, value]) => h("div", null, `${key} ${typeof value === "string" ? value : JSON.stringify(value)}`))],
];

const MEASURES = [
  ["Locations", (s) => s.n],
  ["Average score", (s) => fmt.int(s.average_score)],
  ["Median score", (s) => fmt.int(s.median_score)],
  ["Country rate", (s) => fmt.pct(s.country_success_rate)],
  ["Refusal rate", (s) => fmt.pct(s.refusal_rate)],
  ["Average distance", (s) => fmt.km(s.average_distance_km)],
  ["Median distance", (s) => fmt.km(s.median_distance_km)],
  ["Input tokens", (s) => fmt.int(s.input_tokens)],
  ["Output tokens", (s) => fmt.int(s.output_tokens)],
  ["Cost", (s) => fmt.usd(s.cost)],
];

export function runsView(ctx, view) {
  const block = (name, fields) => [
    h("tr", { class: "group" }, h("th", { colspan: ctx.runs.length + 1 }, name)),
    fields.map(([label, value]) => h("tr", null, h("th", { scope: "row" }, label), ctx.runs.map(({ summary, card }) => h("td", null, value(summary, card))))),
  ];
  view.replaceChildren(
    h("div", { class: "table-wrap" },
      h("table", { class: "runs" },
        h("thead", null, h("tr", null, h("th"), ctx.runs.map(({ card }, i) =>
          h("th", { title: card.run },
            h("span", { class: "column" }, swatch(ctx.runColor(i)), ctx.labels.get(card.run)),
            h("button", { class: "remove", title: "Remove", onclick: () => writeState({ runs: ctx.state.runs.filter((run) => run !== card.run) }) }, "×"),
          )),
        )),
        h("tbody", null, block("Settings", SETTINGS), block("Results", MEASURES)),
      ),
    ),
  );
}
