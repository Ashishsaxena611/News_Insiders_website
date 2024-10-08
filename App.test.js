import { render, screen } from '@testing-library/react';
import App from './App';

test('renders learn react a', () => {
  render(<App />);
  const aElement = screen.getByText(/learn react/i);
  expect(aElement).toBeInTheDocument();
});
