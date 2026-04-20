def test_login_returns_real_token_and_auth_me_resolves_user(client):
    login_response = client.post(
        "/auth/login",
        json={
            "username": "admin",
            "password": "Admin123*",
        },
    )

    assert login_response.status_code == 200
    token_body = login_response.json()

    assert token_body["token_type"] == "bearer"
    assert token_body["username"] == "admin"
    assert token_body["access_token"]
    assert token_body["expires_in"] == 3600

    me_response = client.get(
        "/auth/me",
        headers={
            "Authorization": f"Bearer {token_body['access_token']}",
        },
    )

    assert me_response.status_code == 200
    me_body = me_response.json()
    assert me_body["username"] == "admin"
    assert "id" in me_body


def test_login_returns_401_for_invalid_password(client):
    response = client.post(
        "/auth/login",
        json={
            "username": "admin",
            "password": "mal-password",
        },
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Credenciales inválidas."