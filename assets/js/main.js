(() => {
  // Theme switch
  const root = document.body;
  const themeSwitch = document.getElementById("mood");
  const themeData = root.getAttribute("data-theme");

  if (themeSwitch) {
    initTheme(localStorage.getItem("theme"));
    themeSwitch.addEventListener("click", () =>
      toggleTheme(localStorage.getItem("theme"))
    );

    function toggleTheme(state) {
      var utterances = document.querySelector('iframe');
      if (state === "dark") {
        localStorage.setItem("theme", "light");
        root.removeAttribute("data-theme");
        utterances.contentWindow.postMessage(
          {type: 'set-theme',theme: 'github-light'},
          'https://utteranc.es'
        );
      } else if (state === "light") {
        localStorage.setItem("theme", "dark");
        document.body.setAttribute("data-theme", "dark");
        utterances.contentWindow.postMessage(
          {type: 'set-theme',theme: 'github-dark'},
          'https://utteranc.es'
        );
      } else {
        initTheme(state);
      }
    }

    function initTheme(state) {
      if (state === "dark") {
        document.body.setAttribute("data-theme", "dark");
      } else if (state === "light") {
        root.removeAttribute("data-theme");
      } else {
        localStorage.setItem("theme", themeData);
      }
    }
  }
})();
