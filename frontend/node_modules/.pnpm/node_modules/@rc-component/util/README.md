<div align="center">
  <h1>@rc-component/util</h1>
  <p><sub><a href="https://ant.design"><img alt="Ant Design" height="14" src="https://gw.alipayobjects.com/zos/rmsportal/KDpgvguMpGfqaHPjicRK.svg" style="vertical-align: -0.125em;" /></a> Part of the Ant Design ecosystem.</sub></p>
  <p>🛠️ Shared React utilities for rc-component packages.</p>

  <p>
    <a href="https://npmjs.org/package/@rc-component/util"><img alt="NPM version" src="https://img.shields.io/npm/v/@rc-component/util.svg?style=flat-square"></a>
    <a href="https://npmjs.org/package/@rc-component/util"><img alt="npm downloads" src="https://img.shields.io/npm/dm/@rc-component/util.svg?style=flat-square"></a>
    <a href="https://github.com/react-component/util/actions/workflows/react-component-ci.yml"><img alt="build status" src="https://github.com/react-component/util/actions/workflows/react-component-ci.yml/badge.svg"></a>
    <a href="https://app.codecov.io/gh/react-component/util"><img alt="Codecov" src="https://img.shields.io/codecov/c/github/react-component/util/master.svg?style=flat-square"></a>
    <a href="https://bundlephobia.com/package/@rc-component/util"><img alt="bundle size" src="https://img.shields.io/bundlephobia/minzip/@rc-component/util?style=flat-square"></a>
    <a href="https://github.com/umijs/dumi"><img alt="dumi" src="https://img.shields.io/badge/docs%20by-dumi-blue?style=flat-square"></a>
  </p>
</div>

<p align="center">English | <a href="./README.zh-CN.md">简体中文</a></p>

## Highlights

- Built for React and maintained by the rc-component team.
- Used by Ant Design and other React component libraries.
- Ships TypeScript declarations with both ES module and CommonJS outputs.
- Keeps examples, tests, and preview builds aligned with the package source.

> This package was historically published as `rc-util`. Prefer the scoped package for new code.

## Install

```bash
npm install @rc-component/util
```

## Usage

```tsx | pure
import {
  Portal,
  raf,
  useEvent,
  useLayoutEffect,
  warning,
} from '@rc-component/util';
```

## Examples

Run the local dumi site:

```bash
npm install
npm start
```

Then open `http://localhost:8000`.

## API

The package exposes small, focused helpers used across rc-component and Ant Design packages. Import the public entry when possible, and use subpath imports for specialized DOM, React, or test helpers.

| Area            | Examples                                                                           |
| --------------- | ---------------------------------------------------------------------------------- |
| React hooks     | `useEvent`, `useLayoutEffect`, `useMergedState`, `useState` helpers                |
| React utilities | `Children/toArray`, `composeRef`, `pickAttrs`, `Portal`                            |
| DOM utilities   | `Dom/canUseDom`, `Dom/contains`, `Dom/dynamicCSS`, `Dom/focus`, `getScrollBarSize` |
| Async helpers   | `raf`, `raf.cancel`                                                                |
| Warnings        | `warning`, `noteOnce`, `resetWarned`                                               |
| Tests           | `test/domHook` for DOM prototype spies                                             |

Examples in `docs/examples` cover the commonly used subpath helpers.

## Development

```bash
npm install
npm start
npm test
npm run build
```

The dumi site runs at `http://localhost:8000` by default.

## Release

```bash
npm run prepublishOnly
```

The release flow is handled by `@rc-component/np` through the `rc-np` command after the package build.

## License

@rc-component/util is released under the [MIT](./LICENSE) license.
