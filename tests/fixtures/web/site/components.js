// Fixture web components: open, nested open, closed shadow roots, iframe inside shadow.
customElements.define('login-card', class extends HTMLElement {
  constructor() { super(); const r = this.attachShadow({mode: 'open'});
    r.innerHTML = `<style>:host{display:block;border:1px solid #888;padding:8px}</style>
      <h2>Account</h2><label>User <input id="user" name="user"></label>
      <label>Password <input id="pass" type="password" name="pass"></label>
      <button type="button">Sign in</button><p id="login-result" role="status"></p>`;
    r.querySelector('button').onclick = () => { r.querySelector('#login-result').textContent = 'Welcome, ' + r.querySelector('#user').value; };
  }
});
customElements.define('inner-widget', class extends HTMLElement {
  constructor() { super(); const r = this.attachShadow({mode: 'open'});
    r.innerHTML = `<button type="button" class="deep">Deep action</button><span class="deep-status"></span>`;
    r.querySelector('button').onclick = () => { r.querySelector('.deep-status').textContent = 'deep ok'; };
  }
});
customElements.define('outer-panel', class extends HTMLElement {
  constructor() { super(); this.attachShadow({mode: 'open'}).innerHTML = `<section><h3>Panel</h3><inner-widget></inner-widget></section>`; }
});
customElements.define('closed-counter', class extends HTMLElement {
  constructor() { super(); const r = this.attachShadow({mode: 'closed'});
    r.innerHTML = `<button type="button" aria-label="Increment">+1</button> Count: <output aria-label="Count value">0</output>`;
    r.querySelector('button').onclick = () => { const o = r.querySelector('output'); o.textContent = String(+o.textContent + 1); };
  }
});
customElements.define('shadow-frame', class extends HTMLElement {
  constructor() { super(); this.attachShadow({mode: 'open'}).innerHTML = `<iframe id="couponframe" title="Coupon" src="coupon.html" width="420" height="90"></iframe>`; }
});
customElements.define('card-input', class extends HTMLElement {
  constructor() { super(); const r = this.attachShadow({mode: 'open'});
    r.innerHTML = `<input id="cardno" aria-label="Card number"><button type="button">Verify card</button><b id="card-status"></b>`;
    r.querySelector('button').onclick = () => { r.querySelector('#card-status').textContent = 'card ' + r.querySelector('#cardno').value.slice(-4) + ' verified'; };
  }
});
