document.addEventListener("DOMContentLoaded", () => {
    // 1. Counter Animations
    const counters = document.querySelectorAll('.counter');
    const animationSpeed = 50; 

    setTimeout(() => {
        counters.forEach(counter => {
            const targetStr = counter.getAttribute('data-target');
            const target = +targetStr;
            
            const updateCount = () => {
                const currentRaw = counter.innerHTML.split('<')[0];
                const count = +currentRaw;
                
                const inc = target / animationSpeed;

                if (count < target) {
                    const newVal = Math.min(count + inc, target);
                    counter.innerHTML = newVal.toFixed(2) + '<span class="percent">%</span>';
                    setTimeout(updateCount, 25);
                } else {
                    counter.innerHTML = target.toFixed(2) + '<span class="percent">%</span>';
                }
            };
            updateCount();
        });
    }, 800); // Wait for the initial CSS entry animation to finish

    // 2. Intersection Observer to organically fade-in cards when scrolling downwards
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = 1;
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, { threshold: 0.1 });

    document.querySelectorAll('.image-card').forEach(card => {
        card.style.opacity = 0;
        card.style.transform = 'translateY(50px)';
        card.style.transition = 'all 0.8s cubic-bezier(0.165, 0.84, 0.44, 1)';
        observer.observe(card);
    });
});
