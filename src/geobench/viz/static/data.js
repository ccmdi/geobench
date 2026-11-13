import { brand } from "./color.js";

const DEFAULTS = { view: "map", rows: "location", by: "run", metric: "points", sort: "-spread", q: "", shelf: "", map: "", lab: "", round: "", order: "number", scope: "all", focus: "" };

export function readState() {
  const params = new URLSearchParams(location.hash.slice(1));
  return { ...Object.fromEntries(Object.entries(DEFAULTS).map(([key, value]) => [key, params.get(key) ?? value])), runs: params.getAll("run") };
}

export function writeState(patch, { push = false } = {}) {
  const next = { ...readState(), ...patch };
  const params = new URLSearchParams(next.runs.map((run) => ["run", run]));
  for (const [key, value] of Object.entries(DEFAULTS)) if (next[key] !== value) params.set(key, next[key]);
  history[push ? "pushState" : "replaceState"](null, "", `#${params}`);
  dispatchEvent(new HashChangeEvent("hashchange"));
}

const json = (url) => fetch(url).then((response) => (response.ok ? response.json() : response.text().then((text) => Promise.reject(new Error(text)))));

export const cards = json("/api/runs").then((list) => new Map(list.map((card) => [card.run, card])));

const loads = new Map();

export function load(card) {
  if (!loads.has(card.run)) {
    const loading = json(`/api/run?run=${encodeURIComponent(card.run)}`).then(({ summary, answers }) => ({
      card,
      summary,
      answers: answers.map((answer) => ({ ...answer, card })),
    }));
    loads.set(card.run, loading.catch((error) => (loads.delete(card.run), Promise.reject(error))));
  }
  return loads.get(card.run);
}

export function imageUrl(answer) {
  return `/api/image?${new URLSearchParams({ map: answer.map, location: answer.location })}`;
}

const DISTINCTIONS = [
  (card) => [card.effort ?? (card.thinking ? "thinking" : null), card.search ? "search" : null],
  (card) => [card.map],
  (card) => [card.started.slice(0, 16).replace("T", " ")],
];

export function runLabels(selected) {
  const parts = new Map(selected.map((card) => [card.run, []]));
  const label = (card) => [card.name, ...parts.get(card.run)].join(" ");
  for (const distinction of DISTINCTIONS) {
    for (const rivals of Map.groupBy(selected, label).values()) {
      if (rivals.length > 1) for (const card of rivals) parts.get(card.run).push(...distinction(card).filter(Boolean));
    }
  }
  return new Map(selected.map((card) => [card.run, parts.get(card.run).length ? `${card.name} (${parts.get(card.run).join(", ")})` : card.name]));
}

function carded(axis) {
  return { ...axis, key: (a) => axis.pick(a.card) };
}

export const AXES = {
  location: { name: "Round", key: (a) => a.task, label: (a) => `#${a.location}`, place: (a) => a.place },
  country: { name: "Country", key: (a) => a.place.name, label: (a) => a.place.name, place: (a) => a.place },
  map: carded({ name: "Map", pick: (card) => card.map, label: (a) => a.map }),
  run: carded({ name: "Run", pick: (card) => card.run, label: (a, labels) => labels.get(a.run) }),
  model: carded({ name: "Model", pick: (card) => card.model, label: (a) => a.card.name }),
  lab: carded({ name: "Lab", pick: (card) => card.lab, label: (a) => a.card.lab_name, brand }),
};

export const ROWS = ["location", "country", "map", "model", "lab", "run"];
export const SERIES = ["run", "model", "lab", "map"];

const LOG_KM = Math.log10(20001);

export const fmt = {
  int: (v) => (v == null ? "-" : Math.round(v).toLocaleString("en-US")),
  km: (v) => (v == null ? "-" : `${v < 10 ? v.toFixed(1) : Math.round(v).toLocaleString("en-US")} km`),
  pct: (v) => (v == null ? "-" : `${(v * 100).toFixed(1)}%`),
  usd: (v) => (v == null ? "-" : `$${v.toFixed(v < 1 ? 3 : 2)}`),
};

export const METRICS = {
  points: { name: "Score", value: (a) => a.points, format: fmt.int, goodness: (v) => v / 5000 },
  km: { name: "Distance", value: (a) => a.distance_km, format: fmt.km, goodness: (v) => 1 - Math.log10(1 + v) / LOG_KM },
  hit: { name: "Country", value: (a) => (a.passed ? 1 : 0), format: fmt.pct, goodness: (v) => v },
};

export function mean(values) {
  const known = values.filter((v) => v != null);
  return known.length ? known.reduce((sum, v) => sum + v, 0) / known.length : null;
}

export function groups(answers, axis, labels) {
  const found = new Map();
  for (const answer of answers) {
    const key = axis.key(answer);
    if (!found.has(key)) found.set(key, { key, label: axis.label(answer, labels), answers: [] });
    found.get(key).answers.push(answer);
  }
  return [...found.values()];
}

export function pivot(answers, rows, series, metric, labels) {
  const columns = groups(answers, series, labels);
  const lines = groups(answers, rows, labels).map((row) => {
    const bySeries = Map.groupBy(row.answers, series.key);
    const cells = columns.map((column) => mean((bySeries.get(column.key) ?? []).map(metric.value)));
    const known = cells.filter((v) => v != null);
    return {
      ...row,
      cells,
      n: new Set(row.answers.map((a) => a.task)).size,
      mean: mean(known),
      spread: known.length > 1 ? Math.max(...known) - Math.min(...known) : null,
    };
  });
  return { columns, lines };
}

export function sorted(lines, sort) {
  const field = sort.replace(/^-/, "");
  const value = { key: (l) => l.key, label: (l) => l.label, n: (l) => l.n, mean: (l) => l.mean, spread: (l) => l.spread }[field] ?? ((l) => l.cells[Number(field.slice(1))]);
  const sign = sort.startsWith("-") ? -1 : 1;
  return lines.toSorted((a, b) => {
    const [x, y] = [value(a), value(b)];
    if (x == null || y == null) return (x == null) - (y == null);
    return sign * (typeof x === "string" ? x.localeCompare(y, undefined, { numeric: true }) : x - y);
  });
}

export const ORDERS = {
  number: { name: "Round number", sort: "key" },
  hardest: { name: "Hardest first", sort: "mean" },
  disputed: { name: "Most disputed", sort: "-spread" },
};

export function rounds(answers, order, labels) {
  return sorted(pivot(answers, AXES.location, AXES.run, METRICS.points, labels).lines, ORDERS[order].sort);
}

export function standings(series) {
  return series.map((s) => ({
    ...s,
    score: mean(s.answers.map(METRICS.points.value)),
    km: mean(s.answers.map(METRICS.km.value)),
    hits: s.answers.filter((a) => a.passed).length,
  }));
}
