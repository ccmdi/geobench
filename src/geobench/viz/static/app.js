import { painter } from "./color.js";
import { AXES, SERIES, cards as listing, groups, load, readState, rounds, runLabels, writeState } from "./data.js";
import { fill, h, select, swatch } from "./dom.js";
import { mapView } from "./map.js";
import { roundPanel, step } from "./round.js";
import { runsView, scoresView, tableControls, tableView } from "./views.js";

const VIEWS = {
  map: { name: "Map", render: mapView, controls: (state) => [seriesControl(state)] },
  table: { name: "Table", render: tableView, controls: tableControls },
  scores: { name: "Scores", render: scoresView, controls: (state) => [seriesControl(state)] },
  runs: { name: "Runs", render: runsView, controls: () => [] },
};
const SHELVES = [["", "All"], ["public", "Public"], ["hidden", "Hidden"], ["incomplete", "Incomplete"]];

const $ = (id) => document.getElementById(id);
let turn = 0;
let latest;

function showPanel(shown) {
  document.body.classList.toggle("has-detail", shown);
  $("detail").hidden = !shown;
  return shown;
}

function seriesControl(state) {
  return h("label", null, "Color by", select(state.by, SERIES.map((key) => [key, AXES[key].name]), (by) => writeState({ by })));
}

function haystack(card) {
  return [card.name, card.model, card.map, card.lab_name, card.provider, card.run, card.effort, card.status, card.started, JSON.stringify(card.settings)].join(" ").toLowerCase();
}

function matches(card, state) {
  const words = state.q.toLowerCase().split(/\s+/).filter(Boolean);
  const text = haystack(card);
  return (!state.shelf || card.status === state.shelf) && (!state.map || card.map === state.map) && (!state.lab || card.lab === state.lab) && words.every((word) => text.includes(word));
}

function toggle(state, run) {
  writeState({ runs: state.runs.includes(run) ? state.runs.filter((r) => r !== run) : [...state.runs, run] });
}

function renderFinder(state, cards, paint) {
  const all = [...cards.values()];
  const search = $("search");
  if (search.value !== state.q) search.value = state.q;
  const maps = [...new Set(all.map((card) => card.map))].sort().map((map) => [map, map]);
  const labs = [...new Map(all.map((card) => [card.lab, card.lab_name]))].sort(([, a], [, b]) => a.localeCompare(b));
  $("facets").replaceChildren(
    h("div", { class: "chips", role: "group", "aria-label": "Shelf" }, SHELVES.map(([key, label]) => h("button", { class: "chip", "aria-pressed": String(state.shelf === key), onclick: () => writeState({ shelf: key }) }, label))),
    select(state.map, [["", "Every map"], ...maps], (map) => writeState({ map })),
    select(state.lab, [["", "Every lab"], ...labs], (lab) => writeState({ lab })),
  );
  const shown = all.filter((card) => matches(card, state)).sort((a, b) => b.started.localeCompare(a.started));
  const picked = state.runs.map((run) => cards.get(run)).filter(Boolean);
  const labels = runLabels(picked);
  const filtered = state.q || state.shelf || state.map || state.lab;
  fill(
    $("selected"),
    picked.length
      ? h("div", { class: "picked-head" }, h("h2", null, `${picked.length} selected`), h("button", { class: "link", onclick: () => writeState({ runs: [], round: "" }) }, "Clear"))
      : null,
    h("ul", { class: "picked", "data-scroll": "picked" }, picked.map((card) => h("li", { title: card.run }, swatch(paint(card)), h("span", { class: "grow" }, labels.get(card.run)), h("button", { class: "remove", title: "Remove", onclick: () => toggle(state, card.run) }, "×")))),
  );
  const fresh = shown.filter((card) => !state.runs.includes(card.run));
  fill(
    $("results-head"),
    h("h2", null, `${shown.length} of ${all.length} runs`),
    filtered && fresh.length ? h("button", { class: "link", onclick: () => writeState({ runs: [...state.runs, ...fresh.map((c) => c.run)] }) }, `Add ${fresh.length}`) : null,
  );
  $("results").replaceChildren(
    ...shown.map((card) => {
      const index = state.runs.indexOf(card.run);
      return h("li", null,
        h("button", { class: "run", "aria-pressed": String(index >= 0), title: card.run, onclick: () => toggle(state, card.run) },
          index >= 0 ? swatch(paint(card)) : h("span", { class: "swatch empty" }),
          h("span", { class: "run-text" },
            h("span", { class: "run-name" }, card.name, card.effort ? h("span", { class: "tag" }, card.effort) : card.thinking ? h("span", { class: "tag" }, "thinking") : null, card.search ? h("span", { class: "tag" }, "search") : null),
            h("span", { class: "run-meta" }, h("span", null, card.map), h("span", null, card.started.slice(0, 10)), card.status === "public" ? null : h("span", { class: `shelf ${card.status}` }, card.status)),
          ),
        ),
      );
    }),
  );
}

function renderTabs(state) {
  $("tabs").replaceChildren(
    ...Object.entries(VIEWS).map(([key, view]) => h("button", { class: "tab", "aria-selected": String(state.view === key), onclick: () => writeState({ view: key }, { push: true }) }, view.name)),
  );
}

function renderLegend(ctx) {
  return ctx.state.view === "runs" ? [] : ctx.series.slice(0, 24).map((s) => h("span", { class: "key" }, swatch(s.color), s.label));
}

async function render() {
  const mine = ++turn;
  const state = readState();
  const cards = await listing;
  const axis = AXES[state.by];
  const paint = painter(axis, cards);
  const cardColor = (card) => paint(axis.pick(card));
  renderFinder(state, cards, cardColor);
  renderTabs(state);
  const selected = state.runs.map((run) => cards.get(run)).filter(Boolean);
  const view = $("view");
  if (!selected.length) {
    $("controls").replaceChildren();
    view.replaceChildren(h("div", { class: "empty" }, h("p", null, "Pick runs on the left to see their guesses.")));
    showPanel(false);
    return;
  }
  const pending = setTimeout(() => view.replaceChildren(h("div", { class: "empty" }, h("p", null, `Loading ${selected.length} runs`))), 120);
  let runs;
  try {
    runs = await Promise.all(selected.map(load));
  } catch (error) {
    clearTimeout(pending);
    if (mine === turn) view.replaceChildren(h("div", { class: "empty" }, h("p", null, `Could not load a run: ${error.message}`)));
    return;
  }
  clearTimeout(pending);
  if (mine !== turn) return;
  const answers = runs.flatMap((run) => run.answers);
  const labels = runLabels(selected);
  const series = groups(answers, axis, labels).map((s) => ({ ...s, color: paint(s.key) }));
  const ctx = { state, cards, selected, runs, answers, labels, series, rounds: rounds(answers, state.order, labels), color: paint, tint: (a) => paint(axis.key(a)), cardColor };
  latest = ctx;
  const current = VIEWS[state.view] ?? VIEWS.map;
  $("controls").replaceChildren(h("div", { class: "control-row" }, current.controls(state)), h("div", { class: "legend" }, renderLegend(ctx)));
  if (showPanel(state.view === "map" || ctx.rounds.some((round) => round.key === state.round))) roundPanel(ctx, $("detail"));
  current.render(ctx, view);
}

$("search").addEventListener("input", (event) => writeState({ q: event.target.value }));
addEventListener("hashchange", render);
addEventListener("keydown", (event) => {
  if (event.target.closest("input, select, textarea") || !latest || event.altKey || event.ctrlKey || event.metaKey) return;
  const round = { ArrowLeft: () => step(latest, -1), ArrowRight: () => step(latest, 1), Escape: () => "" }[event.key];
  if (round) writeState({ round: round() });
});
render();
