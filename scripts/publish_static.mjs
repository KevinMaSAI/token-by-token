import { cp, mkdir, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
const root = resolve(import.meta.dirname, "..");
await mkdir(resolve(root, "docs"), { recursive: true });
await cp(resolve(root, "dist"), resolve(root, "docs"), { recursive: true });
await writeFile(resolve(root, "docs/.nojekyll"), "");
console.log("Production files copied to docs/ for GitHub Pages.");
