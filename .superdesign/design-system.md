# Arcana Design System

## Product Context
**Arcana** is an open-source LLM observability platform for forensic analysis of multi-agent execution traces. It reconstructs agent DAGs, detects communication drift, and monitors task progress with LLM-verified checkpoints.

**Target audience**: AI/ML engineers, platform teams, DevOps engineers building multi-agent systems.

**Aesthetic direction**: Dark, technical, engineering-focused. Inspired by terminal UIs and forensic analysis tools. Warm copper accent against near-black backgrounds. Sophisticated but not flashy.

## Typography

### Fonts
- **Display font**: `'Instrument Serif', Georgia, serif` — Used for headings, hero text, large numbers, logo text. Regular and italic styles available.
- **Body font**: `'DM Sans', system-ui, sans-serif` — Used for body text, labels, buttons, navigation. Weights: 300 (light), 400 (regular), 500 (medium), 600 (semibold).

### Scale
- Hero h1: `3.8rem` (desktop), `2.8rem` (tablet), `2.2rem` (mobile)
- Section titles: `2.4rem` (desktop), `2rem` (tablet), `1.7rem` (mobile)
- CTA heading: `2.8rem` (desktop), `2rem` (mobile)
- Card headings: `1.2-1.25rem` (Instrument Serif, weight 400)
- Body text: `1rem-1.1rem`
- Small text: `.875rem`
- Labels/tags: `.65-.78rem` (uppercase, letter-spacing .04-.15em)
- Code: `.82rem`

### Line Heights
- Headings: `1.08-1.2`
- Body: `1.6-1.7`

## Color Palette

### Core Colors
| Name | Value | CSS Variable | Usage |
|------|-------|-------------|-------|
| Background | `#08080c` | `--bg` | Page/app background |
| Sidebar | `#0c0c12` | `--sidebar` | Dashboard sidebar |
| Card | `#111118` | `--card` | Card backgrounds, panels |
| Card Hover | `#16161f` | `--card-hover` | Card hover state |
| Surface | `#0e0e15` | `--surface` | Elevated surfaces |

### Border Colors
| Name | Value | CSS Variable |
|------|-------|-------------|
| Border | `#1e1e2c` | `--border` |
| Border Light | `#2a2a3c` | `--border-light` |

### Accent Colors
| Name | Value | CSS Variable | Dim Variant |
|------|-------|-------------|-------------|
| Copper (Primary) | `#c8956c` | `--copper` | `rgba(200,149,108,.12)` |
| Green (Success) | `#4ade80` | `--green` | `rgba(74,222,128,.1)` |
| Red (Error/Drift) | `#f87171` | `--red` | `rgba(248,113,113,.1)` |
| Blue (Info) | `#60a5fa` | `--blue` | `rgba(96,165,250,.1)` |
| Purple (Secondary) | `#a78bfa` | `--purple` | `rgba(167,139,250,.1)` |
| Yellow (Warning) | `#fbbf24` | `--yellow` | `rgba(251,191,36,.1)` |

### Text Colors
| Name | Value | CSS Variable | Usage |
|------|-------|-------------|-------|
| Primary | `#e8e4df` | `--text` | Main body text (warm white) |
| Secondary | `#7b7688` | `--text-secondary` | Muted/supporting text |
| Dim | `#4a4558` | `--text-dim` | Labels, captions |

## Spacing

- Container max-width: `1200px` with `24px` horizontal padding
- Section padding: `100px 0` (desktop), `60px 0` (mobile)
- Card padding: `32px` (desktop), `24px` (mobile)
- Grid gaps: `20-32px`
- Component gaps: `8-16px`

## Border Radius
- Default: `8px` (`--radius`)
- Small: `5px` (`--radius-sm`) — inputs
- Large: `12px` (`--radius-lg`) — cards, panels
- Full: `100px` — badges, tags, pills

## Shadows
- Float cards: `0 20px 60px rgba(0,0,0,.4), 0 0 0 1px rgba(255,255,255,.03) inset`
- Float card hover: `0 30px 80px rgba(0,0,0,.5), 0 0 0 1px rgba(200,149,108,.1) inset`
- Button hover: `0 8px 30px rgba(200,149,108,.2)`
- Tilt card hover: `0 30px 60px rgba(0,0,0,.3), 0 0 0 1px rgba(200,149,108,.08) inset`

## Components

### Buttons
- **Primary**: Copper background, dark text, 12px 28px padding, 8px radius, 600 weight
  - Hover: brightness(1.15), translateY(-2px), copper glow shadow
- **Outline**: Transparent bg, white text, 1px border-light border
  - Hover: copper border + text, translateY(-2px)
- **Nav CTA**: Copper bg, dark text, 8px 20px padding, smaller
- All buttons use `inline-flex` with `align-items: center; gap: 8px`

### Cards
- **Feature card**: Card bg, 1px border, 12px radius, top gradient line on hover (copper)
- **Problem card**: Semi-transparent bg `rgba(17,17,24,.5)`, border, translateY(-4px) on hover
- **Float card** (hero): Absolute positioned, deep shadow, 3D transforms, floating animation

### Badges/Tags
- **Hero badge**: Pill shape (100px radius), 6px 16px padding, border, small text
- **Feature tag**: Pill shape, tiny text (.68rem), uppercase, bold, color-coded by accent

### Section Headers
- **Label**: Uppercase, .7rem, copper color, left line decoration, letter-spacing .15em
- **Title**: Instrument Serif, 2.4rem, weight 400
- **Description**: 1rem, text-secondary color, max-width 560px

## Background & Atmosphere
- Multi-layer radial gradients (copper glow top, blue glow right, purple glow left)
- Dot grid overlay (40px spacing, 3% white opacity)
- Scanline overlay (copper-tinted, 8% opacity, 4px repeat)
- Cursor-following glow orb (400px, copper radial gradient)
- Scroll progress bar (copper gradient, 2px height, fixed top)

## Motion & Animation
- Default transition: `all .2s cubic-bezier(.4,0,.2,1)`
- Fade-in on scroll: `opacity .7s + transform .7s` with staggered delays (.1s increments)
- Float animations: 6-8s ease-in-out infinite translateY cycles
- 3D card tilt on hover (perspective 800px, ±5deg rotation)
- Magnetic button pull effect (15% of mouse offset)
- DAG edge drawing animation (stroke-dashoffset, staggered .15s)
- Node stagger pulse (scale 0.7→1, bouncy easing)
- Mini bar growth (scaleY from bottom, staggered)
- Code typewriter effect (character-by-character with cursor blink)
- Step connection line (scaleX from left)
- Step number ripple (box-shadow pulse)
- Animated counters (ease-out cubic, 1200ms)
- All animations respect `prefers-reduced-motion: reduce`

## Responsive Breakpoints
| Breakpoint | Target | Key Changes |
|-----------|--------|-------------|
| `max-width: 767px` | Mobile | Stack all grids to 1-col, reduce hero text, hamburger menu, reduce padding |
| `max-width: 1023px` | Tablet | Hero to 1-col, reduce card sizes, 2-col features |
| `min-width: 1024px` | Desktop | Full 2-col hero, 3-col problems, 2-col features |

## Logo
- Diamond-shaped icon: 28x28px border with 8x8px filled center, rotated 45deg
- Copper border color (#c8956c)
- Text: "Arcana" in Instrument Serif, 1.4rem
- Layout: horizontal flex, icon + text with 10px gap

## Accessibility
- Skip-to-content link (implicit)
- ARIA labels on interactive elements
- Keyboard-navigable hamburger menu
- `prefers-reduced-motion` disables all animations
- Semantic HTML (nav, section, footer, headings hierarchy)
- Min 44px touch targets on mobile
