const GOLDEN = 137.50776;
const dark = matchMedia("(prefers-color-scheme: dark)");
const THEMES = {
  light: { levels: [0.56, 0.66, 0.47], chroma: 0.15 },
  dark: { levels: [0.72, 0.81, 0.63], chroma: 0.13 },
};

const BRANDS = {
  anthropic: ["#c96442", "#e08a6c"],
  openai: ["#202123", "#ececf1"],
  google: ["#1a73e8", "#5e97f6"],
  meta: ["#0866ff", "#4d94ff"],
  mistral: ["#e8590c", "#fa7a3c"],
  xai: ["#6b7280", "#9ca3af"],
  alibaba: ["#615ced", "#8f8bf5"],
  zhipuai: ["#3d5afe", "#7a8cff"],
  deepseek: ["#4d6bfe", "#7d91ff"],
};

function gamma(x) {
  const clamped = Math.max(0, Math.min(1, x));
  return clamped <= 0.0031308 ? 12.92 * clamped : 1.055 * clamped ** (1 / 2.4) - 0.055;
}

function oklch(l, c, h) {
  const [a, b] = [c * Math.cos((h * Math.PI) / 180), c * Math.sin((h * Math.PI) / 180)];
  const [lc, mc, sc] = [l + 0.3963377774 * a + 0.2158037573 * b, l - 0.1055613458 * a - 0.0638541728 * b, l - 0.0894841775 * a - 1.291485548 * b].map((v) => v ** 3);
  const rgb = [
    4.0767416621 * lc - 3.3077115913 * mc + 0.2309699292 * sc,
    -1.2684380046 * lc + 2.6097574011 * mc - 0.3413193965 * sc,
    -0.0041960863 * lc - 0.7034186147 * mc + 1.707614701 * sc,
  ];
  return `#${rgb.map((v) => Math.round(gamma(v) * 255).toString(16).padStart(2, "0")).join("")}`;
}

export function generated(slot) {
  const theme = THEMES[dark.matches ? "dark" : "light"];
  return oklch(theme.levels[slot % theme.levels.length], theme.chroma, (250 + slot * GOLDEN) % 360);
}

export function brand(lab) {
  return BRANDS[lab]?.[dark.matches ? 1 : 0];
}

export function painter(axis, cards) {
  const order = [...new Set([...cards.values()].toSorted((a, b) => a.started.localeCompare(b.started)).map(axis.pick))];
  const slots = new Map(order.map((key, slot) => [key, slot]));
  return (key) => axis.brand?.(key) ?? generated(slots.get(key) ?? order.length);
}
