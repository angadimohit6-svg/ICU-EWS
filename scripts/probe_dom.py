import json
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome", headless=True)
    page = b.new_page(viewport={"width": 1500, "height": 950})
    page.goto("http://localhost:8501", wait_until="networkidle")
    page.wait_for_timeout(2500)
    data = page.evaluate("""() => {
      const sb = document.querySelector('section[data-testid="stSidebar"]') || [...document.querySelectorAll('div,section,aside')].find(e => getComputedStyle(e).position === 'fixed' && e.getBoundingClientRect().width > 200 && e.getBoundingClientRect().left === 0);
      const header = document.querySelector('[data-testid="stHeader"]');
      const main = document.querySelector('[data-testid="stMain"]');
      const block = document.querySelector('[data-testid="stMainBlockContainer"]');
      const pop = document.querySelector('[data-testid="stPopover"]');
      const info = (el) => el ? {tag: el.tagName, cls: el.className, testid: el.getAttribute('data-testid'), rect: el.getBoundingClientRect().toJSON(), pos: getComputedStyle(el).position, bg: getComputedStyle(el).backgroundColor, w: getComputedStyle(el).width} : null;
      return {sb: info(sb), sbParent: sb ? info(sb.parentElement) : null, header: info(header), main: info(main), block: info(block), pop: info(pop),
        appChildren: [...document.querySelector('.stApp').children].map(e => info(e)),
        sbChildren: sb ? [...sb.children].map(e => ({tag: e.tagName, testid: e.getAttribute('data-testid'), cls: e.className.slice(0,100)})) : null,
        sbHTML: sb ? sb.outerHTML.slice(0, 700) : null};
    }""")
    print(json.dumps(data, indent=1)[:4000])
    b.close()
