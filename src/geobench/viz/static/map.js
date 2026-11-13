import * as L from "https://unpkg.com/leaflet@1.9.4/dist/leaflet-src.esm.js";
import { AXES, fmt, groups, writeState } from "./data.js";
import { h } from "./dom.js";

const element = document.createElement("div");
element.id = "map";
let map, layer, framed;

function nearest(lng, origin) {
  return lng + 360 * Math.round((origin - lng) / 360);
}

function open(task) {
  writeState({ loc: task }, { push: true });
}

export function mapView(ctx, view) {
  view.replaceChildren(element);
  if (!map) {
    map = L.map(element, { preferCanvas: true, worldCopyJump: true, zoomSnap: 0.5 }).setView([20, 0], 2);
    L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 18, attribution: "&copy; OpenStreetMap contributors" }).addTo(map);
  }
  map.invalidateSize();
  layer?.remove();
  layer = L.layerGroup().addTo(map);
  const css = getComputedStyle(document.documentElement);
  const [ink, paper] = ["--ink", "--panel"].map((name) => css.getPropertyValue(name).trim());
  const focus = ctx.state.loc;
  const truths = [];
  const focused = [];
  const locations = groups(ctx.answers, AXES.location);
  for (const location of locations) {
    const { lat, lng, location: id, place } = location.answers[0];
    const lit = !focus || location.key === focus;
    for (const answer of location.answers.filter((a) => a.guess_lat != null)) {
      const guess = [answer.guess_lat, nearest(answer.guess_lng, lng)];
      const color = ctx.tint(answer);
      const tip = () => h("div", null, h("strong", null, ctx.labels.get(answer.run)), h("br"), `${answer.guess_country ?? "No country"}, ${fmt.km(answer.distance_km)}, ${fmt.int(answer.points)} pts`);
      L.polyline([[lat, lng], guess], { color, weight: location.key === focus ? 3 : 1.5, opacity: lit ? 0.8 : 0.08 })
        .bindTooltip(tip, { sticky: true })
        .on("click", () => open(location.key))
        .addTo(layer);
      L.circleMarker(guess, { radius: lit ? 3.5 : 2, stroke: false, fillColor: color, fillOpacity: lit ? 1 : 0.15 }).addTo(layer);
      if (location.key === focus) focused.push(guess);
    }
    truths.push([lat, lng]);
    if (location.key === focus) focused.push([lat, lng]);
    L.circleMarker([lat, lng], { radius: location.key === focus ? 7 : 4.5, color: ink, weight: 2, opacity: lit ? 1 : 0.3, fillColor: paper, fillOpacity: lit ? 1 : 0.3 })
      .bindTooltip(() => h("span", null, `#${id} ${place}`))
      .on("click", () => open(location.key))
      .addTo(layer);
  }
  const frame = focus || ctx.selected.map((card) => card.run).join("|");
  if (frame !== framed && truths.length) {
    framed = frame;
    map.fitBounds(focus && focused.length ? focused : truths, { padding: [48, 48], maxZoom: focus ? 7 : 4 });
  }
}
