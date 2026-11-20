// The product name is a setting (M3 spec §6.11): LOOM_BRAND_NAME, read on the server each
// time it is asked for (never a NEXT_PUBLIC_ variable, never at build time), so one image
// serves any name. Unset or blank: no name is shown, and each page uses its own title and
// a neutral header.

type Env = Record<string, string | undefined>;

export function brandName(env: Env = process.env): string | undefined {
  const name = env.LOOM_BRAND_NAME?.trim();
  return name ? name : undefined;
}
