const SVG = "http://www.w3.org/2000/svg";

export function fill(element, ...children) {
  element.replaceChildren(...children.flat(Infinity).filter((child) => child != null && child !== false));
  return element;
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
  return fill(element, ...children);
}

export function select(value, options, onchange) {
  return h("select", { onchange: (event) => onchange(event.target.value) }, options.map(([key, label]) => h("option", { value: key, selected: key === value }, label)));
}

export function swatch(color) {
  return h("span", { class: "swatch", style: `--c: ${color}` });
}
