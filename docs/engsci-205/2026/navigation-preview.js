// Native <details> gives keyboard-accessible, script-independent disclosure.
// This only adds current-location highlighting and opens a deep-linked group.
document.addEventListener('DOMContentLoaded', () => {
  const toc = document.querySelector('#quarto-margin-sidebar > #TOC.engsci205-toc');
  if (!toc) return;
  const groups = [...toc.querySelectorAll('details[data-sections]')];
  const sections = groups.flatMap((group, index) =>
    group.dataset.sections.split(' ').map(id => ({id, group: index, element: document.getElementById(id)}))
  ).filter(section => section.element);
  let pending = false;

  function markLocation(openHashGroup = false) {
    const hash = decodeURIComponent(location.hash.slice(1));
    let current = sections[0];
    for (const section of sections) {
      if (section.element.getBoundingClientRect().top <= 175) current = section;
      else break;
    }
    if (openHashGroup && hash) current = sections.find(section => section.id === hash) || current;
    if (!current) return;
    // Quarto can clone the margin menu on narrow/overlapped layouts. Update
    // any copies too; native disclosure and ordinary anchors still work there.
    document.querySelectorAll('.engsci205-toc').forEach(menu => {
      const menuGroups = [...menu.querySelectorAll('details[data-sections]')];
      menuGroups.forEach((group, index) => {
        const active = index === current.group;
        group.classList.toggle('is-current', active);
        if (openHashGroup && hash && active) group.open = true;
        const represented = [...group.querySelectorAll('a[data-section]')];
        const members = group.dataset.sections.split(' ');
        const currentIndex = members.indexOf(current.id);
        let selected = null;
        if (active) {
          for (const link of represented) {
            if (members.indexOf(link.dataset.section) <= currentIndex) selected = link;
          }
        }
        represented.forEach(link => {
          const isCurrent = link === selected;
          link.classList.toggle('is-current', isCurrent);
          if (isCurrent) link.setAttribute('aria-current', 'location');
          else link.removeAttribute('aria-current');
        });
      });
    });
  }

  function scheduleUpdate() {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => { pending = false; markLocation(); });
  }
  document.addEventListener('scroll', scheduleUpdate, {passive: true});
  window.addEventListener('resize', scheduleUpdate);
  window.addEventListener('hashchange', () => markLocation(true));
  window.addEventListener('load', () => markLocation(true));
  markLocation(true);
});
