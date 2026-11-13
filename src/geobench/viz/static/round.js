import { AXES, METRICS, ORDERS, fmt, groups, imageUrl, mean, standings, writeState } from "./data.js";
import { fill, flag, h, reveal, select, swatch } from "./dom.js";
import { MEASURES, SETTINGS } from "./views.js";

const photo = h("img", { alt: "" });
let showing = "";

function stat(name, value, title) {
  return h("div", null, h("dt", null, name), h("dd", { class: "num", title }, value));
}

function tint(value) {
  return value == null ? null : `--g: ${(METRICS.points.goodness(value)).toFixed(3)}`;
}

export function step(ctx, by) {
  const order = ctx.rounds;
  const index = order.findIndex((round) => round.key === ctx.state.round);
  if (!order.length) return "";
  if (index < 0) return order[by > 0 ? 0 : order.length - 1].key;
  return order[(index + by + order.length) % order.length].key;
}

function picture(url) {
  if (url !== showing) {
    showing = url;
    const next = new Image();
    next.src = url;
    next.decode().catch(() => null).then(() => showing === url && (photo.src = url));
  }
  return h("a", { class: "photo", href: url, target: "_blank" }, photo);
}

function preload(ctx) {
  for (const by of [-1, 1]) {
    const round = ctx.rounds.find((r) => r.key === step(ctx, by));
    if (round) new Image().src = imageUrl(round.answers[0]);
  }
}

function focus(run, label) {
  return h("button", { class: "run-link", title: run, onclick: () => writeState({ focus: run }) }, label);
}

function runCard(ctx) {
  const run = ctx.runs.find(({ card }) => card.run === ctx.state.focus);
  if (!run) return null;
  const facts = (fields) => fields.map(([label, value]) => h("div", null, h("dt", null, label), h("dd", null, value(run.summary, run.card))));
  return h("section", { class: "run-card" },
    h("header", null,
      swatch(ctx.cardColor(run.card)),
      h("h3", { class: "grow", title: run.card.run }, ctx.labels.get(run.card.run)),
      h("button", { class: "remove", title: "Close", onclick: () => writeState({ focus: "" }) }, "×"),
    ),
    h("dl", null, facts(SETTINGS), facts(MEASURES)),
  );
}

function guessRow(ctx, answer) {
  return h("li", null,
    h("div", { class: "standing" },
      swatch(ctx.tint(answer)),
      h("span", { class: "grow" }, focus(answer.run, ctx.labels.get(answer.run))),
      answer.refused ? h("span", { class: "muted" }, "Refused") : flag(answer.guess_place, `${answer.guess_place?.name ?? "No country"}, ${answer.passed ? "right" : "wrong"} country`),
      h("span", { class: "num muted" }, fmt.km(answer.distance_km)),
      h("span", { class: `num points ${answer.passed ? "hit" : "miss"}` }, fmt.int(answer.points)),
    ),
    answer.reasoning ? h("details", null, h("summary", null, "Reasoning"), h("pre", null, answer.reasoning)) : null,
    answer.reply ? h("details", null, h("summary", null, "Reply"), h("pre", null, answer.reply)) : answer.error ? h("p", { class: "muted" }, answer.error) : null,
  );
}

function seriesRow(ctx, standing) {
  return h("li", null,
    h("div", { class: "standing" },
      swatch(standing.color),
      h("span", { class: "grow" }, ctx.state.by === "run" ? focus(standing.key, standing.label) : standing.label),
      h("span", { class: "num muted", title: "Right country" }, `${standing.hits}/${standing.answers.length}`),
      h("span", { class: "num muted" }, fmt.km(standing.km)),
      h("span", { class: "num points" }, fmt.int(standing.score)),
    ),
  );
}

export function roundPanel(ctx, panel) {
  const { state } = ctx;
  const active = ctx.rounds.find((round) => round.key === state.round);
  const scoped = active ? active.answers : ctx.answers;
  const table = standings(groups(scoped, AXES[state.by], ctx.labels)).map((s) => ({ ...s, color: ctx.color(s.key) }));
  const scored = table.filter((s) => s.score != null).toSorted((a, b) => b.score - a.score);
  const [best, worst] = [scored[0], scored.at(-1)];
  const go = (round) => writeState({ round });
  const truth = active?.answers[0];
  const maps = new Set(ctx.answers.map((a) => a.map)).size > 1;
  fill(panel,
    h("header", { class: "round-head" },
      h("div", { class: "round-title" },
        truth ? flag(truth.place) : null,
        h("h2", null, truth ? `Round ${truth.location}` : "All rounds"),
        truth && maps ? h("span", { class: "muted" }, truth.map) : null,
      ),
      h("nav", { class: "round-nav" },
        h("button", { class: "icon", title: "Previous round", onclick: () => go(step(ctx, -1)) }, "‹"),
        h("button", { class: "icon", title: "Next round", onclick: () => go(step(ctx, 1)) }, "›"),
        h("button", { class: "chip", "aria-pressed": String(!active), title: "All rounds", onclick: () => go("") }, "All"),
      ),
    ),
    runCard(ctx),
    h("div", { class: "round-order" }, select(state.order, Object.entries(ORDERS).map(([key, order]) => [key, order.name]), (order) => writeState({ order }))),
    h("ol", { class: "strip", "data-scroll": "strip" }, ctx.rounds.map((round) => {
      const first = round.answers[0];
      return h("li", null,
        h("button", {
          class: "round-cell cell",
          "aria-current": round.key === state.round ? "true" : null,
          style: tint(round.mean),
          title: `${maps ? `${first.map} ` : ""}#${first.location} ${first.place.name}, mean ${fmt.int(round.mean)}`,
          onclick: () => go(round.key),
        }, first.location),
      );
    })),
    h("dl", { class: "stats" },
      stat("Mean", fmt.int(mean(table.map((s) => s.score)))),
      stat("Best", fmt.int(best?.score), best?.label),
      stat("Worst", fmt.int(worst?.score), worst?.label),
      stat("Country", `${scoped.filter((a) => a.passed).length}/${scoped.length}`),
    ),
    truth ? picture(imageUrl(truth)) : null,
    h("ol", { class: "standings" }, active ? active.answers.toSorted((a, b) => (b.points ?? -1) - (a.points ?? -1)).map((a) => guessRow(ctx, a)) : table.toSorted((a, b) => (b.score ?? -1) - (a.score ?? -1)).map((s) => seriesRow(ctx, s))),
  );
  reveal(panel.querySelector(".round-cell[aria-current]"));
  if (active) preload(ctx);
}
