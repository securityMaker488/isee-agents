import { describe, it, expect, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { App } from '../App';
import {
  getApiBaseUrl,
  getStoredToken,
  setStoredToken,
} from '../services/api';

describe('I.See Agents Frontend Architecture & Auth Suite', () => {
  beforeEach(() => {
    setStoredToken(null);
  });

  it('stores and clears JWT token in localStorage', () => {
    expect(getStoredToken()).toBeNull();
    setStoredToken('isee-test-jwt-token-2026');
    expect(getStoredToken()).toBe('isee-test-jwt-token-2026');
    setStoredToken(null);
    expect(getStoredToken()).toBeNull();
  });

  it('resolves default FastAPI v1 endpoint URL', () => {
    const url = getApiBaseUrl();
    expect(url).toContain('/api/v1');
  });

  it('renders I.See Technologies Authentication Screen when unauthenticated', async () => {
    render(<App />);
    expect(await screen.findByText(/I\.See Agents/i)).toBeInTheDocument();
    expect(
      screen.getByText(/Controle de Acesso JWT/i)
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Acesso Rápido com Credencial Admin Padrão/i)
    ).toBeInTheDocument();
  });
});
