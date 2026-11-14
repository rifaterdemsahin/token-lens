// Shared: language (EN/TR) persisted in localStorage; toggles <html lang>.
(function () {
  let l = (navigator.language || "").startsWith("tr") ? "tr" : "en";
  try { l = localStorage.getItem("tl-lang") || l; } catch (e) {}
  document.documentElement.lang = l;
  window.TL_LANG = () => document.documentElement.lang;
  window.TL_SETLANG = v => { document.documentElement.lang = v; try { localStorage.setItem("tl-lang", v); } catch (e) {}
    document.dispatchEvent(new CustomEvent("tl-lang", { detail: v })); };
  document.addEventListener("DOMContentLoaded", () => {
    const b = document.getElementById("langbtn");
    if (b) { const sync = () => b.textContent = TL_LANG() === "en" ? "TR" : "EN";
      b.onclick = () => { TL_SETLANG(TL_LANG() === "en" ? "tr" : "en"); sync(); }; sync(); }
    const here = location.pathname.replace(/\/$/, "") || "/";
    document.querySelectorAll("nav.top a").forEach(a => { if (a.getAttribute("href") === here) a.classList.add("cur"); });
  });
})();
