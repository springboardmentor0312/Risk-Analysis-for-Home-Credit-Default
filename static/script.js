function validateForm() {
    let isValid = true;
    const form = document.getElementById('creditForm');
    const inputs = form.querySelectorAll('input, select');

    inputs.forEach(input => {
        if (input.value.trim() === '') {
            showError(input, 'This field is required');
            isValid = false;
        } else {
            removeError(input);
        }
    });

    return isValid;
}

function showError(input, message) {
    const parent = input.parentElement;
    let error = parent.querySelector('.error');
    if (!error) {
        error = document.createElement('div');
        error.classList.add('error');
        parent.appendChild(error);
    }
    error.textContent = message;
}

function removeError(input) {
    const parent = input.parentElement;
    const error = parent.querySelector('.error');
    if (error) {
        error.remove();
    }
}
