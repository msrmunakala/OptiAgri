"""Backend smoke tests for OptiCrop routes."""

from app import app


def main() -> None:
    sample_payload = {
        "N": "90",
        "P": "42",
        "K": "43",
        "temperature": "24.8",
        "humidity": "82",
        "ph": "6.5",
        "rainfall": "210",
    }

    with app.test_client() as client:
        checks = [
            ("GET /", client.get("/")),
            ("GET /about", client.get("/about")),
            ("GET /dashboard", client.get("/dashboard")),
            ("POST /predict", client.post("/predict", data=sample_payload)),
            ("POST /predict invalid", client.post("/predict", data={**sample_payload, "ph": "99"})),
        ]

        for name, response in checks:
            expected = 400 if "invalid" in name else 200
            assert response.status_code == expected, f"{name} returned {response.status_code}"
            print(f"{name}: {response.status_code}")


if __name__ == "__main__":
    main()
