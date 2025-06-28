module.exports = {
  presets: [
    [
      'next/babel',
      {
        'preset-react': {
          runtime: 'automatic',
          importSource: '@emotion/react',
        },
      },
    ],
    ['@babel/preset-typescript'],
  ],
  plugins: [
    '@babel/plugin-proposal-class-properties',
    '@babel/plugin-syntax-dynamic-import',
    '@babel/plugin-transform-runtime',
  ],
  env: {
    test: {
      presets: [
        [
          'next/babel',
          {
            'preset-react': {
              runtime: 'automatic',
              importSource: '@emotion/react',
            },
          },
        ],
        ['@babel/preset-typescript'],
      ],
      plugins: [
        '@babel/plugin-transform-modules-commonjs',
        'babel-plugin-dynamic-import-node',
      ],
    },
  },
};
