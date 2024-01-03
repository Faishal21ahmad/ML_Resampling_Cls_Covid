function openDropdown(idbtn, idlist) {
    const list = document.getElementById(idlist);
    list.classList.remove('hidden');
}

function closeDropDown(idbtn, idlist, namelist, idtxtbtn) {
    const list = document.getElementById(idlist);
    list.classList.add('hidden');

    const txtbtn = document.getElementById(idtxtbtn);
    txtbtn.innerText = namelist;
}

document.addEventListener('DOMContentLoaded', function () {
    setTimeout(function () {
        // Menambahkan kelas 'hidden' setelah 3 detik
        document.getElementById('alert').classList.add('hidden');
    }, 3000);
});