function addCopyButtonsToCodeBlocks() {
    document.querySelectorAll('pre').forEach(block => {
        const wrapper = document.createElement('div');
        wrapper.className = 'code-block';
        block.parentNode.insertBefore(wrapper, block);
        wrapper.appendChild(block);
        
        const button = document.createElement('button');
        button.className = 'copy-button';
        button.innerHTML = '<i class="fas fa-copy"></i>';
        
        button.addEventListener('click', async () => {
            const code = block.querySelector('code');
            await navigator.clipboard.writeText(code.innerText);
            
            button.innerHTML = '<i class="fas fa-check"></i>';
            setTimeout(() => {
                button.innerHTML = '<i class="fas fa-copy"></i>';
            }, 2000);
        });
        
        wrapper.appendChild(button);
    });
}

function setupSmoothScrolling() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            const targetElement = document.querySelector(targetId);
            
            if (targetElement) {
                const navHeight = document.querySelector('nav').offsetHeight;
                const targetPosition = targetElement.offsetTop - navHeight;
                
                window.scrollTo({
                    top: targetPosition,
                    behavior: 'smooth'
                });
            }
        });
    });
}

document.addEventListener('DOMContentLoaded', () => {
    addCopyButtonsToCodeBlocks();
    setupSmoothScrolling();
});
