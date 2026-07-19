const sampleValues = {
    N: 90,
    P: 42,
    K: 43,
    temperature: 24.8,
    humidity: 82,
    ph: 6.5,
    rainfall: 210
};

document.addEventListener("DOMContentLoaded", () => {
    document.body.classList.add("is-ready");
    const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    const sampleButton = document.getElementById("sampleInputBtn");
    if (sampleButton) {
        sampleButton.addEventListener("click", () => {
            Object.entries(sampleValues).forEach(([name, value]) => {
                const input = document.querySelector(`[name="${name}"]`);
                if (input) {
                    input.value = value;
                    input.dispatchEvent(new Event("input", { bubbles: true }));
                }
            });
        });
    }

    document.querySelectorAll(".input-shell input").forEach((input) => {
        input.addEventListener("input", () => {
            input.closest(".input-shell")?.classList.toggle("has-value", input.value.trim() !== "");
        });
        input.dispatchEvent(new Event("input"));
    });

    document.querySelectorAll(".needs-validation").forEach((form) => {
        form.addEventListener("submit", (event) => {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add("was-validated");
        });
    });

    const revealItems = document.querySelectorAll(".reveal-on-scroll");
    if (prefersReducedMotion || !("IntersectionObserver" in window)) {
        revealItems.forEach((item) => item.classList.add("is-visible"));
    } else {
        const revealObserver = new IntersectionObserver((entries) => {
            entries.forEach((entry) => {
                if (entry.isIntersecting) {
                    entry.target.classList.add("is-visible");
                    revealObserver.unobserve(entry.target);
                }
            });
        }, { threshold: 0.16 });

        revealItems.forEach((item, index) => {
            item.style.transitionDelay = `${Math.min(index * 45, 240)}ms`;
            revealObserver.observe(item);
        });
    }

    const counters = document.querySelectorAll("[data-count-to]");
    counters.forEach((counter) => {
        const target = Number.parseFloat(counter.dataset.countTo || "0");
        const isPercent = counter.textContent.includes("%");
        if (prefersReducedMotion || Number.isNaN(target)) {
            counter.textContent = isPercent ? `${target.toFixed(2)}%` : Math.round(target).toLocaleString();
            return;
        }

        const startTime = performance.now();
        const duration = 900;
        const render = (now) => {
            const progress = Math.min((now - startTime) / duration, 1);
            const eased = 1 - Math.pow(1 - progress, 3);
            const value = target * eased;
            counter.textContent = isPercent ? `${value.toFixed(2)}%` : Math.round(value).toLocaleString();
            if (progress < 1) requestAnimationFrame(render);
        };
        requestAnimationFrame(render);
    });
});
