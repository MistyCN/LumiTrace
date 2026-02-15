(() => {
  const coachForm = document.getElementById("coachForm");
  const submitBtn = document.getElementById("coachSubmitBtn");
  const coachHint = document.getElementById("coachHint");
  const scrollTarget = document.body?.dataset?.scrollTarget || "";

  if (scrollTarget === "coach") {
    const coachSection = document.getElementById("coach-section");
    if (coachSection) {
      coachSection.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  if (!coachForm) return;
  coachForm.addEventListener("submit", () => {
    if (coachHint) coachHint.textContent = "正在生成，请稍候...";
    if (submitBtn) {
      submitBtn.disabled = true;
      submitBtn.textContent = "生成中...";
    }
  });
})();
