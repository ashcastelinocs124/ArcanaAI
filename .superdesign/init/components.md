# Shared UI Components

## Framework
- **Tech stack**: Vanilla HTML/CSS/JS (no framework, no build system)
- **CSS**: Inline `<style>` block in single HTML files
- **Fonts**: DM Sans (body) + Instrument Serif (display)

## Component Primitives

All components are CSS classes defined inline in each HTML file. No shared component library.

### Buttons
```css
.btn-primary {
  background: var(--copper);
  color: var(--bg);
  padding: 12px 28px;
  border-radius: var(--radius);
  font-size: .9rem;
  font-weight: 600;
  transition: var(--transition);
  display: inline-flex;
  align-items: center;
  gap: 8px;
}
.btn-primary:hover { filter: brightness(1.15); transform: translateY(-2px); box-shadow: 0 8px 30px rgba(200,149,108,.2) }

.btn-outline {
  background: transparent;
  color: var(--text);
  padding: 12px 28px;
  border-radius: var(--radius);
  font-size: .9rem;
  font-weight: 500;
  border: 1px solid var(--border-light);
  transition: var(--transition);
  display: inline-flex;
  align-items: center;
  gap: 8px;
}
.btn-outline:hover { border-color: var(--copper); color: var(--copper); transform: translateY(-2px) }
```

### Cards
```css
.float-card {
  position: absolute;
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  overflow: hidden;
  box-shadow: 0 20px 60px rgba(0,0,0,.4), 0 0 0 1px rgba(255,255,255,.03) inset;
}

.problem-card {
  padding: 32px 28px;
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  background: rgba(17,17,24,.5);
  transition: var(--transition);
}

.feature-card {
  padding: 32px;
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  position: relative;
  overflow: hidden;
}
```

### Badges / Tags
```css
.hero-badge {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 6px 16px;
  border: 1px solid var(--border);
  border-radius: 100px;
  font-size: .78rem;
  color: var(--text-secondary);
  background: rgba(17,17,24,.6);
}

.feature-tag {
  display: inline-flex;
  margin-top: 16px;
  padding: 3px 10px;
  border-radius: 100px;
  font-size: .68rem;
  font-weight: 600;
  letter-spacing: .04em;
  text-transform: uppercase;
}
```
