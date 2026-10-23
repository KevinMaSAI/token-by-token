import { test, expect } from "@playwright/test";
test("手算路径、对照生成、空输入、符号字符与界面留档", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/");
  await expect(
    page.locator("[data-hero] .pill").filter({ hasText: "计算完成" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "进入逐步手算" }).click();
  await page.getByRole("button", { name: "4. 改一个数，再继续" }).click();
  await page.getByLabel("手算中间值").fill("1.2");
  await page.getByRole("button", { name: "从这里继续" }).click();
  await expect(page.getByRole("table", { name: "概率差值热图" })).toBeVisible();
  await page.getByRole("button", { name: "还原", exact: true }).click();
  await page.getByRole("button", { name: "关闭逐步手算" }).click();
  await page.getByRole("navigation").getByRole("button").nth(5).click();
  await expect(
    page.getByRole("heading", { name: "注意力全景 · 每一层，每一个头" }),
  ).toBeVisible();
  await page.screenshot({
    path: "docs/attention-preview.png",
    fullPage: false,
  });
  await page.getByRole("navigation").getByRole("button").nth(9).click();
  await page.getByRole("button", { name: "运行两组对比" }).click();
  await expect(page.getByRole("heading", { name: /配置 B/ })).toBeVisible({
    timeout: 30000,
  });
  await page.getByLabel("输入文字").fill("🙂 你好\n12!");
  await expect(
    page.locator("[data-hero] .pill").filter({ hasText: "计算完成" }),
  ).toBeVisible();
  await page.getByRole("navigation").getByRole("button").nth(0).click();
  await expect(
    page.getByRole("heading", { name: "计算机先读到的是字节" }),
  ).toBeVisible();
  await page.getByLabel("输入文字").fill("");
  await expect(
    page.locator("[data-hero] .pill").filter({ hasText: "计算完成" }),
  ).toBeVisible();
  await expect(page.getByText("（空输入：模型使用起始标记）")).toBeVisible();
  expect(errors).toEqual([]);
});
test("500 步完整语料训练、下载再导入、离线推理与移动布局", async ({ page }) => {
  await page.goto("/");
  await expect(
    page.locator("[data-hero] .pill").filter({ hasText: "计算完成" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "训练一个模型" }).click();
  await page.getByRole("button", { name: "开始真实训练" }).click();
  await expect(
    page.getByRole("status").filter({ hasText: "训练完成" }),
  ).toBeVisible({ timeout: 60000 });
  const labels = await page.locator("main .delta-i").allTextContents();
  console.log("500-step training:", labels.join(" | "));
  test.info().annotations.push({
    type: "training-observation",
    description: labels.join(" | "),
  });
  const promise = page.waitForEvent("download");
  await page.getByRole("button", { name: "导出训练权重 JSON" }).click();
  const download = await promise;
  const path = await download.path();
  expect(path).toBeTruthy();
  await page.getByLabel("模型来源").selectOption("mini");
  await expect(page.getByLabel("模型来源")).toHaveValue("mini");
  await page.locator("input[type=file]").setInputFiles(path!);
  await expect(page.getByLabel("模型来源")).toHaveValue("trained");
  await page.context().setOffline(true);
  await page.getByRole("navigation").getByRole("button").nth(5).click();
  await page.getByRole("button", { name: "生成一步", exact: true }).click();
  await expect(page.getByRole("button", { name: "回放第 1 步" })).toBeVisible();
  await page.context().setOffline(false);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({ path: "docs/mobile-preview.png", fullPage: false });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth + 1,
    ),
  ).toBe(true);
});
