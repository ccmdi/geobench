import { AXES, SERIES, cards as listing, fmt, groups, imageUrl, load, readState, runLabels, writeState } from "./data.js";
import { fill, h, select, swatch } from "./dom.js";
import { mapView } from "./map.js";
import { runsView, scoresView, tableControls, tableView } from "./views.js";

const VIEWS = {
  map: { name: "Map", render: mapView, controls: (state) => [seriesControl(state)] },
  table: { name: "Table", render: tableView, controls: tableControls },
  scores: { name: "Scores", render: scoresView, controls: (state) => [seriesControl(state)] },
  runs: { name: "Runs", render: runsView, controls: () => [] },
};
const SHELVES = [["", "All"], ["public", "Public"], ["hidden", "Hidden"], ["incomplete", "Incomplete"]];
const PALETTE = 8;

const $ = (id) => document.getElementById(id);
let turn = 0;

function seriesControl(state) {
  return h("label", null, "Color by", select(state.by, SERIES.map((key) => [key, AXES[key].name]), (by) => writeState({ by })));
}

function palette() {
  const css = getComputedStyle(document.documentElement);
  return (i) => css.getPropertyValue(i < PALETTE ? `--s${i + 1}` : "--other").trim();
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

function renderFinder(state, cards, color) {
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
      ? h("div", { class: "picked-head" }, h("h2", null, `${picked.length} selected`), h("button", { class: "link", onclick: () => writeState({ runs: [], loc: "" }) }, "Clear"))
      : null,
    h("ul", { class: "picked" }, picked.map((card, i) => h("li", { title: card.run }, swatch(color(i)), h("span", { class: "grow" }, labels.get(card.run)), h("button", { class: "remove", title: "Remove", onclick: () => toggle(state, card.run) }, "×")))),
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
          index >= 0 ? swatch(color(index)) : h("span", { class: "swatch empty" }),
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

function renderDetail(ctx) {
  const detail = $("detail");
  const answers = ctx.answers.filter((a) => a.task === ctx.state.loc);
  document.body.classList.toggle("has-detail", answers.length > 0);
  detail.hidden = !answers.length;
  if (!answers.length) return;
  const truth = answers[0];
  const order = groups(ctx.answers, AXES.location).map((g) => g.key).sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
  const step = (by) => order[(order.indexOf(truth.task) + by + order.length) % order.length];
  const ranked = answers.toSorted((a, b) => (b.points ?? -1) - (a.points ?? -1));
  detail.replaceChildren(
    h("header", { class: "detail-head" },
      h("div", null, h("h2", null, truth.place), h("p", { class: "muted" }, `${truth.map} #${truth.location}`)),
      h("nav", null,
        h("button", { class: "icon", title: "Previous location", onclick: () => writeState({ loc: step(-1) }) }, "‹"),
        h("button", { class: "icon", title: "Next location", onclick: () => writeState({ loc: step(1) }) }, "›"),
        h("button", { class: "icon", title: "Close", onclick: () => writeState({ loc: "" }) }, "×"),
      ),
    ),
    h("a", { class: "photo", href: imageUrl(truth), target: "_blank" }, h("img", { src: imageUrl(truth), alt: `Location ${truth.location}` })),
    h("p", { class: "truth" }, h("span", { class: "muted" }, "Truth "), truth.country, h("span", { class: "num muted" }, ` ${fmt.coord(truth.lat, truth.lng)}`)),
    h("ol", { class: "guesses" }, ranked.map((a) =>
      h("li", null,
        h("div", { class: "guess-head" },
          swatch(ctx.tint(a)),
          h("span", { class: "grow" }, ctx.labels.get(a.run)),
          h("span", { class: "num points" }, a.refused ? "Refused" : fmt.int(a.points)),
        ),
        a.refused
          ? h("p", { class: "guess muted" }, a.error ?? "No answer")
          : h("p", { class: "guess" },
              h("span", { class: a.passed ? "hit" : "miss", title: a.passed ? "Right country" : "Wrong country" }, a.guess_country ?? "No country"),
              h("span", { class: "num muted" }, ` ${fmt.coord(a.guess_lat, a.guess_lng)}, ${fmt.km(a.distance_km)} off`),
            ),
        a.reasoning ? h("details", null, h("summary", null, "Reasoning"), h("pre", null, a.reasoning)) : null,
        a.reply ? h("details", null, h("summary", null, "Reply"), h("pre", null, a.reply)) : null,
      ),
    )),
  );
}

async function render() {
  const mine = ++turn;
  const state = readState();
  const cards = await listing;
  const color = palette();
  renderFinder(state, cards, color);
  renderTabs(state);
  const selected = state.runs.map((run) => cards.get(run)).filter(Boolean);
  const view = $("view");
  if (!selected.length) {
    $("controls").replaceChildren();
    view.replaceChildren(h("div", { class: "empty" }, h("p", null, "Pick runs on the left to see their guesses.")));
    renderDetail({ state, answers: [] });
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
  const axis = AXES[state.by];
  const series = groups(answers, axis, labels).map((s, i) => ({ ...s, color: color(i) }));
  const colors = new Map(series.map((s) => [s.key, s.color]));
  const ctx = { state, cards, selected, runs, answers, labels, series, color: (key) => colors.get(key), tint: (a) => colors.get(axis.key(a)), runColor: color };
  const current = VIEWS[state.view] ?? VIEWS.map;
  $("controls").replaceChildren(h("div", { class: "control-row" }, current.controls(state)), h("div", { class: "legend" }, renderLegend(ctx)));
  renderDetail(ctx);
  current.render(ctx, view);
}

$("search").addEventListener("input", (event) => writeState({ q: event.target.value }));
addEventListener("hashchange", render);
addEventListener("keydown", (event) => {
  if (event.target.closest("input, select, textarea") || !readState().loc) return;
  const button = { Escape: "Close", ArrowLeft: "Previous location", ArrowRight: "Next location" }[event.key];
  if (button) document.querySelector(`#detail button[title="${button}"]`)?.click();
});
render();
