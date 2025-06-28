# Human Voice AI Frontend

A Next.js application that provides a voice interface for interacting with AI. This project allows users to record their voice, select an emotion, and get AI-powered feedback on their speech patterns.

## Features

- Voice recording with real-time visualization
- Emotion selection for voice modulation
- Responsive design for all device sizes
- Built with Next.js 14 and React 19
- TypeScript for type safety
- Tailwind CSS for styling
- Comprehensive test coverage

## Getting Started

### Prerequisites

- Node.js 18.x or later
- npm 9.x or later

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/jadenfix/HumanVoiceAI.git
   cd HumanVoiceAI/human-voice-ai-frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Create a `.env.local` file in the project root and add your environment variables:
   ```env
   NEXT_PUBLIC_APP_ENV=development
   # Add other environment variables here
   ```

### Development

Start the development server:

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser to see the application.

## Testing

This project uses Jest and React Testing Library for testing. To run the tests:

```bash
# Run all tests
npm test

# Run tests in watch mode
npm run test:watch

# Run tests with coverage
npm run test:coverage

# Run tests in CI mode
npm run test:ci
```

### Testing Guidelines

- Unit tests should be placed in `__tests__` directories next to the components they test
- Test files should be named with the pattern `*.test.tsx` or `*.test.ts`
- Use `@testing-library/react` for component testing
- Mock external dependencies and APIs
- Follow the Arrange-Act-Assert pattern

## Linting and Type Checking

```bash
# Run ESLint
npm run lint

# Run TypeScript type checking
npm run type-check

# Fix linting issues
npm run lint:fix
```

## Building for Production

```bash
# Build the application
npm run build

# Start the production server
npm start
```

## Deployment

This project is configured for deployment on Vercel. The deployment is automated via GitHub Actions.

### Environment Variables

The following environment variables need to be set in your Vercel project:

- `NEXT_PUBLIC_APP_ENV` - The environment (e.g., 'development', 'staging', 'production')
- Add other required environment variables here

### Manual Deployment

1. Install Vercel CLI:
   ```bash
   npm install -g vercel
   ```

2. Deploy to Vercel:
   ```bash
   vercel
   ```

### Automated Deployment

Push to the `main` branch to trigger an automatic deployment to production. Pushes to other branches will create preview deployments.

## Project Structure

```
/human-voice-ai-frontend
├── .github/               # GitHub Actions workflows
├── __mocks__/             # Jest mocks
├── public/                # Static files
├── src/
│   ├── app/               # Next.js app directory
│   │   ├── __tests__/     # Test files
│   │   ├── layout.tsx     # Root layout
│   │   └── page.tsx       # Home page
│   ├── components/        # Reusable components
│   ├── hooks/             # Custom React hooks
│   ├── styles/            # Global styles
│   └── utils/             # Utility functions
├── .eslintrc.json         # ESLint configuration
├── .gitignore
├── jest.config.js         # Jest configuration
├── next.config.js         # Next.js configuration
├── package.json
├── postcss.config.js      # PostCSS configuration
├── README.md
├── tailwind.config.js     # Tailwind CSS configuration
└── tsconfig.json          # TypeScript configuration
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

Jaden Fix - [GitHub](https://github.com/jadenfix) | [LinkedIn](https://linkedin.com/in/jadenfix) | jadenfix123@gmail.com
