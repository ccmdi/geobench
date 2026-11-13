const SVG = "http://www.w3.org/2000/svg";

function present(children) {
  return children.flat(Infinity).filter((child) => child != null && child !== false);
}

export function fill(element, ...children) {
  const scrolled = new Map([...element.querySelectorAll("[data-scroll]")].map((kept) => [kept.dataset.scroll, kept.scrollTop]));
  element.replaceChildren(...present(children));
  for (const kept of element.querySelectorAll("[data-scroll]")) kept.scrollTop = scrolled.get(kept.dataset.scroll) ?? 0;
  return element;
}

export function reveal(item) {
  const box = item?.offsetParent;
  if (!box) return;
  const [top, bottom] = [item.offsetTop, item.offsetTop + item.offsetHeight];
  if (top < box.scrollTop || bottom > box.scrollTop + box.clientHeight) box.scrollTop = top - (box.clientHeight - item.offsetHeight) / 2;
}

export function h(tag, props, ...children) {
  const [space, name] = tag.startsWith("svg:") ? [SVG, tag.slice(4)] : [null, tag];
  const element = space ? document.createElementNS(space, name) : document.createElement(name);
  for (const [key, value] of Object.entries(props ?? {})) {
    if (value == null || value === false) continue;
    if (key.startsWith("on")) element.addEventListener(key.slice(2), value);
    else if (key in element && !space && typeof value !== "string") element[key] = value;
    else element.setAttribute(key, value === true ? "" : value);
  }
  element.append(...present(children));
  return element;
}

export function select(value, options, onchange) {
  return h("select", { onchange: (event) => onchange(event.target.value) }, options.map(([key, label]) => h("option", { value: key, selected: key === value }, label)));
}

const FLAGS = "https://cdn.jsdelivr.net/npm/flag-icons@7.6.0/flags/4x3";

export function flag(place, title = place?.name) {
  if (!place) return h("span", { class: "flag none", title: "No country" }, "-");
  return place.code ? h("img", { class: "flag", src: `${FLAGS}/${place.code}.svg`, alt: place.name, title }) : h("span", { class: "flag named", title }, place.name);
}

export function swatch(color) {
  return h("span", { class: "swatch", style: `--c: ${color}` });
}
