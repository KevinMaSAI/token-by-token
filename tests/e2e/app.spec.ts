import { test, expect } from "@playwright/test";
test("十个阶段、真实乘加、分词回放与因果遮罩", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/");
  await expect(
    page.locator("[data-hero] .pill").filter({ hasText: "计算完成" }),
  ).toBeVisible();
  await expect(
    page.getByRole("navigation", { name: "处理阶段" }).getByRole("button"),
  ).toHaveCount(10);
  await expect(
    page.getByRole("heading", { name: "计算机先读到的是字节" }),
  ).toBeVisible();
  await page
    .getByRole("navigation")
    .getByRole("button", { name: "1 分词", exact: true })
    .click();
  await page.getByRole("button", { name: "下一步", exact: true }).click();
  await expect(page.getByText("第 1 / 8 轮")).toBeVisible();
  await page
    .getByRole("navigation")
    .getByRole("button", { name: "3 嵌入", exact: true })
    .click();
  await expect(
    page.getByRole("table", { name: "嵌入矩阵 E", exact: true }),
  ).toBeVisible();
  await page.getByLabel("嵌入新值").fill("0.5");
  await page.getByRole("button", { name: "修改并重新计算" }).click();
  await expect(
    page.locator("[data-hero] .pill").filter({ hasText: "计算完成" }),
  ).toBeVisible();
  for (const i of [2, 4, 5, 6, 7, 8, 9]) {
    await page.getByRole("navigation").getByRole("button").nth(i).click();
    await expect(page.locator("main .card").first()).toBeVisible();
  }
  await expect(page.getByText("max =", { exact: false }).first()).toBeVisible();
  await page.getByRole("navigation").getByRole("button").nth(5).click();
  await expect(
    page.getByRole("table", { name: /层 \d 头 \d 注意力/ }),
  ).toHaveCount(4);
  await expect(page.locator(".att td[data-masked=true]").first()).toBeVisible();
  await expect(page.locator("[data-arithmetic-term]").first()).toBeVisible();
  await page.getByRole("switch", { name: "因果遮罩：不看未来" }).click();
  await expect(page.locator(".att td[data-masked=true]")).toHaveCount(0);
  expect(errors).toEqual([]);
});
test("逐词生成、完整时间线恢复与重新分支", async ({ page }) => {
  await page.goto("/");
  await expect(
    page.locator("[data-hero] .pill").filter({ hasText: "计算完成" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "生成一步", exact: true }).click();
  await expect(page.getByRole("button", { name: "回放第 1 步" })).toBeVisible();
  await expect(
    page.locator("[data-hero] .pill").filter({ hasText: "计算完成" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "生成一步", exact: true }).click();
  await expect(page.getByRole("button", { name: "回放第 2 步" })).toBeVisible();
  await page.getByRole("button", { name: "回放第 1 步" }).click();
  await expect(page.getByText("回放第 1 步：", { exact: false })).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "输出投影：为词表里的每一个候选打分" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "生成一步", exact: true }).click();
  await expect(
    page.locator("[data-hero] .pill").filter({ hasText: "计算完成" }),
  ).toBeVisible();
  await expect(page.getByRole("button", { name: "回放第 2 步" })).toHaveCount(
    0,
  );
});
test("任意中间张量扰动、恢复、真实训练与 JSON 下载", async ({ page }) => {
  await page.goto("/");
  await expect(
    page.locator("[data-hero] .pill").filter({ hasText: "计算完成" }),
  ).toBeVisible();
  await page.getByLabel("扰动张量").selectOption("L0.qRaw");
  await page.getByLabel("扰动行").fill("3");
  await page.getByLabel("扰动幅度").fill("2");
  await page.getByRole("button", { name: "施加扰动" }).click();
  await expect(page.getByRole("table", { name: "概率差值热图" })).toBeVisible();
  await page.getByRole("button", { name: "还原", exact: true }).click();
  await expect(page.getByRole("table", { name: "概率差值热图" })).toHaveCount(
    0,
  );
  await page.getByRole("button", { name: "训练一个模型" }).click();
  await page.getByLabel("训练步数").fill("50");
  await page.getByRole("button", { name: "开始真实训练" }).click();
  await expect(
    page.getByRole("status").filter({ hasText: "训练完成" }),
  ).toBeVisible({ timeout: 60000 });
  await expect(
    page.getByRole("table", { name: "训练中的嵌入权重" }),
  ).toBeVisible();
  await expect(page.getByLabel("模型来源")).toHaveValue("trained");
  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: "导出训练权重 JSON" }).click();
  expect((await download).suggestedFilename()).toBe(
    "token-by-token-trained.json",
  );
});
test("真实模型缺失时清晰降级，窄屏无页面溢出", async ({ page }) => {
  await page.goto("/");
  await expect(
    page.locator("[data-hero] .pill").filter({ hasText: "计算完成" }),
  ).toBeVisible();
  await page.getByLabel("模型来源").selectOption("real");
  await expect(page.getByRole("alert")).toContainText("没有找到预训练文件");
  await expect(
    page.getByRole("heading", { name: "计算机先读到的是字节" }),
  ).toBeVisible();
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.getByLabel("输入文字")).toBeVisible();
  const dimensions = await page.evaluate(() => ({
    width: document.documentElement.scrollWidth,
    viewport: innerWidth,
  }));
  expect(dimensions.width).toBeLessThanOrEqual(dimensions.viewport + 1);
  await page.getByRole("navigation").getByRole("button").nth(5).click();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth + 1,
    ),
  ).toBe(true);
});
