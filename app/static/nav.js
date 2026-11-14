// Shared: language (EN/TR) persisted in localStorage; toggles <html lang>.
(function () {
  let l = (navigator.language || "").startsWith("tr") ? "tr" : "en";
  try { l = localStorage.getItem("tl-lang") || l; } catch (e) {}
  document.documentElement.lang = l;
  window.TL_LANG = () => document.documentElement.lang;
  window.TL_SETLANG = v => { document.documentElement.lang = v; try { localStorage.setItem("tl-lang", v); } catch (e) {}
    document.dispatchEvent(new CustomEvent("tl-lang", { detail: v })); };
  document.addEventListener("DOMContentLoaded", () => {
    const btns = document.querySelectorAll(".flags button");
    const sync = () => btns.forEach(x => x.setAttribute("aria-pressed", String(x.dataset.l === TL_LANG())));
    btns.forEach(x => x.onclick = () => { TL_SETLANG(x.dataset.l); sync(); }); sync();
    const here = location.pathname.replace(/\/$/, "") || "/";
    document.querySelectorAll("nav.top a").forEach(a => { if (a.getAttribute("href") === here) a.classList.add("cur"); });
  });
})();
