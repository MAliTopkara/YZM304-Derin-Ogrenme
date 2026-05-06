/** Frontend public/sample_images/ altındaki örnek görseller. */

export interface SampleImage {
  class: string;
  slug: string;
  /** public/ kökünden relative path. */
  file: string;
}

export const SAMPLE_IMAGES: SampleImage[] = [
  { class: "Among Us",       slug: "among_us",       file: "/sample_images/among_us.png" },
  { class: "Apex Legends",   slug: "apex_legends",   file: "/sample_images/apex_legends.png" },
  { class: "Fortnite",       slug: "fortnite",       file: "/sample_images/fortnite.png" },
  { class: "Forza Horizon",  slug: "forza_horizon",  file: "/sample_images/forza_horizon.png" },
  { class: "Free Fire",      slug: "free_fire",      file: "/sample_images/free_fire.png" },
  { class: "Genshin Impact", slug: "genshin_impact", file: "/sample_images/genshin_impact.png" },
  { class: "God of War",     slug: "god_of_war",     file: "/sample_images/god_of_war.png" },
  { class: "Minecraft",      slug: "minecraft",      file: "/sample_images/minecraft.png" },
  { class: "Roblox",         slug: "roblox",         file: "/sample_images/roblox.png" },
  { class: "Terraria",       slug: "terraria",       file: "/sample_images/terraria.png" },
];

/** /sample_images/foo.png → public URL'den File objesine dönüştür. */
export async function fetchSampleAsFile(sample: SampleImage): Promise<File> {
  const res = await fetch(sample.file);
  if (!res.ok) {
    throw new Error(`Örnek görsel yüklenemedi: ${sample.file}`);
  }
  const blob = await res.blob();
  const filename = `${sample.slug}.png`;
  return new File([blob], filename, { type: blob.type || "image/png" });
}
