import { fmt, writeState } from "./data.js";
import { flag, h } from "./dom.js";

const { MapboxOverlay, LineLayer, ScatterplotLayer } = globalThis.deck;
const STYLES = { light: "https://tiles.openfreemap.org/styles/positron", dark: "https://tiles.openfreemap.org/styles/dark" };
const dark = matchMedia("(prefers-color-scheme: dark)");
const still = matchMedia("(prefers-reduced-motion: reduce)");

const element = h("div", { id: "map" });
const TIP = { backgroundColor: "var(--panel)", color: "var(--ink)", padding: "8px 10px", border: "1px solid var(--rule)", borderRadius: "6px", fontSize: "13px" };
let map, overlay, framed;

function rgb(css) {
  const hex = css.replace("#", "");
  return [0, 2, 4].map((i) => parseInt(hex.slice(i, i + 2), 16));
}

function nearest(lng, origin) {
  return lng + 360 * Math.round((origin - lng) / 360);
}

function tip(content) {
  return { html: h("div", null, content).innerHTML, className: "map-tip", style: TIP };
}

function create() {
  map = new maplibregl.Map({ container: element, style: STYLES[dark.matches ? "dark" : "light"], center: [10, 20], zoom: 1.2, keyboard: false, attributionControl: { compact: true } });
  map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-left");
  overlay = new MapboxOverlay({
    interleaved: false,
    getTooltip: ({ object }) => object?.tip(),
    onHover: ({ object }) => (map.getCanvas().style.cursor = object ? "pointer" : ""),
    onClick: ({ object }) => object?.open(),
  });
  map.addControl(overlay);
  dark.addEventListener("change", () => map.setStyle(STYLES[dark.matches ? "dark" : "light"]));
}

export function mapView(ctx, view) {
  view.replaceChildren(element);
  if (!map) create();
  map.resize();
  const css = getComputedStyle(document.documentElement);
  const [ink, paper] = ["--ink", "--panel"].map((name) => rgb(css.getPropertyValue(name).trim()));
  const { round, focus } = ctx.state;
  const shown = round ? ctx.answers.filter((a) => a.task === round) : ctx.answers;
  const truths = [...Map.groupBy(shown, (a) => a.task).values()].map(([a]) => ({
    at: [a.lng, a.lat],
    open: () => writeState({ round: a.task }, { push: true }),
    tip: () => tip([flag(a.place), h("strong", null, a.place.name), h("span", { class: "muted" }, ` #${a.location}`)]),
  }));
  const guesses = shown
    .filter((a) => a.guess_lat != null)
    .map((a) => ({
      from: [a.lng, a.lat],
      at: [nearest(a.guess_lng, a.lng), a.guess_lat],
      color: [...rgb(ctx.tint(a)), !focus || a.run === focus ? 230 : 50],
      open: () => writeState({ round: a.task, focus: a.run }, { push: true }),
      tip: () => tip([
        h("strong", null, ctx.labels.get(a.run)),
        h("span", { class: "tip-line" }, flag(a.guess_place), a.guess_place?.name ?? "No country"),
        h("span", { class: "tip-line num" }, `${fmt.int(a.points)} pts, ${fmt.km(a.distance_km)}`),
      ]),
    }));
  overlay.setProps({
    layers: [
      new LineLayer({ id: "lines", data: guesses, pickable: true, getSourcePosition: (g) => g.from, getTargetPosition: (g) => g.at, getColor: (g) => g.color, getWidth: round ? 3 : 1.5, widthUnits: "pixels" }),
      new ScatterplotLayer({ id: "guesses", data: guesses, pickable: true, getPosition: (g) => g.at, getFillColor: (g) => g.color, getLineColor: paper, stroked: true, lineWidthUnits: "pixels", getLineWidth: 1.5, radiusUnits: "pixels", getRadius: round ? 7 : 4.5 }),
      new ScatterplotLayer({ id: "truths", data: truths, pickable: true, getPosition: (t) => t.at, getFillColor: paper, getLineColor: ink, stroked: true, lineWidthUnits: "pixels", getLineWidth: 2.5, radiusUnits: "pixels", getRadius: round ? 9 : 5 }),
    ],
  });
  const frame = round || ctx.selected.map((card) => card.run).join("|");
  const points = [...truths.map((t) => t.at), ...(round ? guesses.map((g) => g.at) : [])];
  if (frame !== framed && points.length) {
    framed = frame;
    const [lngs, lats] = [points.map((p) => p[0]), points.map((p) => p[1])];
    const bounds = [[Math.min(...lngs), Math.min(...lats)], [Math.max(...lngs), Math.max(...lats)]];
    map.fitBounds(bounds, { padding: 64, maxZoom: round ? 11 : 3.5, duration: still.matches ? 0 : 900 });
  }
}
