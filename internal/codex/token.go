// Package codex handles Codex/OpenAI OAuth token management
// and Responses API integration for ChatGPT subscription access.
package codex

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

const (
	codexAuthPath = ".codex/auth.json"
	issuer        = "https://auth.openai.com"
	clientID      = "app_EMoamEEZ73f0CkXaXp7hrann"
)

type codexAuth struct {
	AuthMode   string          `json:"auth_mode"`
	Tokens     codexTokens     `json:"tokens"`
	LastRefresh string         `json:"last_refresh"`
}

type codexTokens struct {
	IDToken      string `json:"id_token"`
	AccessToken  string `json:"access_token"`
	RefreshToken string `json:"refresh_token"`
	AccountID    string `json:"account_id"`
}

// jwtPayload is a minimal JWT payload for exp extraction.
type jwtPayload struct {
	Exp int64 `json:"exp"`
}

// CodexCredentials holds the current valid credentials for Codex API calls.
type CodexCredentials struct {
	AccessToken string
	AccountID   string
}

var (
	mu           sync.Mutex
	lastTokens   *codexAuth
	lastModTime  time.Time
)

// GetCredentials reads Codex OAuth credentials from ~/.codex/auth.json,
// refreshing the access token if expired. Results are cached per file mtime.
func GetCredentials() (*CodexCredentials, error) {
	mu.Lock()
	defer mu.Unlock()

	home, err := os.UserHomeDir()
	if err != nil {
		return nil, fmt.Errorf("home dir: %w", err)
	}
	path := filepath.Join(home, codexAuthPath)

	stat, err := os.Stat(path)
	if err != nil {
		return nil, fmt.Errorf("codex auth file: %w", err)
	}

	modTime := stat.ModTime()
	if lastTokens != nil && modTime.Equal(lastModTime) {
		return &CodexCredentials{
			AccessToken: lastTokens.Tokens.AccessToken,
			AccountID:   lastTokens.Tokens.AccountID,
		}, nil
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read codex auth: %w", err)
	}

	var auth codexAuth
	if err := json.Unmarshal(data, &auth); err != nil {
		return nil, fmt.Errorf("parse codex auth: %w", err)
	}
	lastTokens = &auth
	lastModTime = modTime

	if auth.Tokens.AccessToken == "" {
		return nil, fmt.Errorf("no access_token in codex auth")
	}

	exp, err := extractExp(auth.Tokens.AccessToken)
	if err == nil && exp > 0 && time.Until(time.Unix(exp, 0)) < 5*time.Minute {
		refreshed, err := refresh(auth.Tokens.RefreshToken)
		if err != nil {
			return nil, fmt.Errorf("token refresh: %w", err)
		}
		auth.Tokens.AccessToken = refreshed.AccessToken
		auth.Tokens.RefreshToken = refreshed.RefreshToken
		auth.Tokens.AccountID = refreshed.AccountID
		auth.LastRefresh = time.Now().UTC().Format(time.RFC3339Nano)
		if writeErr := writeAuth(path, &auth); writeErr == nil {
			lastTokens = &auth
			lastModTime = time.Now()
		}
	}

	return &CodexCredentials{
		AccessToken: auth.Tokens.AccessToken,
		AccountID:   auth.Tokens.AccountID,
	}, nil
}

func extractExp(token string) (int64, error) {
	parts := strings.Split(token, ".")
	if len(parts) != 3 {
		return 0, fmt.Errorf("not a JWT")
	}
	decoded, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		// Try padding
		decoded, err = base64.RawStdEncoding.DecodeString(parts[1])
		if err != nil {
			return 0, fmt.Errorf("decode JWT payload: %w", err)
		}
	}
	var p jwtPayload
	if err := json.Unmarshal(decoded, &p); err != nil {
		return 0, err
	}
	return p.Exp, nil
}

type tokenResponse struct {
	IDToken      string `json:"id_token"`
	AccessToken  string `json:"access_token"`
	RefreshToken string `json:"refresh_token"`
	ExpiresIn    int    `json:"expires_in"`
	AccountID    string
}

func refresh(refreshToken string) (*tokenResponse, error) {
	form := url.Values{
		"grant_type":    {"refresh_token"},
		"refresh_token": {refreshToken},
		"client_id":     {clientID},
	}
	resp, err := http.PostForm(issuer+"/oauth/token", form)
	if err != nil {
		return nil, fmt.Errorf("refresh request: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("refresh failed %d: %s", resp.StatusCode, string(body))
	}
	var tr tokenResponse
	if err := json.NewDecoder(resp.Body).Decode(&tr); err != nil {
		return nil, fmt.Errorf("decode refresh: %w", err)
	}
	// Extract account_id from id_token if available
	var accountID string
	if tr.IDToken != "" {
		if claims, err := extractAccountID(tr.IDToken); err == nil {
			accountID = claims
		}
	}
	if accountID == "" && tr.AccessToken != "" {
		if claims, err := extractAccountID(tr.AccessToken); err == nil {
			accountID = claims
		}
	}
	tr.IDToken = accountID
	tr.AccountID = accountID
	return &tr, nil
}

type idTokenClaims struct {
	ChatGPTAccountID string `json:"chatgpt_account_id"`
	Auth             *struct {
		ChatGPTAccountID string `json:"chatgpt_account_id"`
	} `json:"https://api.openai.com/auth"`
}

func extractAccountID(token string) (string, error) {
	parts := strings.Split(token, ".")
	if len(parts) != 3 {
		return "", fmt.Errorf("not a JWT")
	}
	decoded, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		decoded, err = base64.RawStdEncoding.DecodeString(parts[1])
		if err != nil {
			return "", err
		}
	}
	var claims idTokenClaims
	if err := json.Unmarshal(decoded, &claims); err != nil {
		return "", err
	}
	if claims.ChatGPTAccountID != "" {
		return claims.ChatGPTAccountID, nil
	}
	if claims.Auth != nil && claims.Auth.ChatGPTAccountID != "" {
		return claims.Auth.ChatGPTAccountID, nil
	}
	return "", fmt.Errorf("no account_id in token")
}

func writeAuth(path string, auth *codexAuth) error {
	data, err := json.MarshalIndent(auth, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0600)
}
