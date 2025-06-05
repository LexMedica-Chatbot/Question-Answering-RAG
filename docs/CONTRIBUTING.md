# Contributing Guide

Terima kasih atas minat Anda untuk berkontribusi pada Question Answering RAG System!

## 🚀 Quick Start untuk Developer

### Setup Development Environment

1. **Fork dan clone repository:**

    ```bash
    git clone https://github.com/yourusername/question-answering-rag.git
    cd question-answering-rag
    ```

2. **Setup virtual environment:**

    ```bash
    make setup-env
    # Windows
    qa-system\Scripts\activate
    # Linux/Mac
    source qa-system/bin/activate
    ```

3. **Install development dependencies:**
    ```bash
    make install-dev
    ```

## 📁 Project Structure

```
src/
├── api/          # API endpoints dan server
├── ingestion/    # Data ingestion dan preprocessing
├── processing/   # Data processing dan transformation
└── demo/         # Demo applications
```

## 🔧 Development Workflow

### Code Style

Project ini menggunakan:

-   **Black** untuk code formatting
-   **isort** untuk import sorting
-   **flake8** untuk linting
-   **mypy** untuk type checking

Jalankan formatting sebelum commit:

```bash
make format
make lint
```

### Testing

```bash
# Run all tests
make test

# Run specific test file
pytest tests/test_api.py -v
```

### Menambahkan Fitur Baru

1. **Buat branch baru:**

    ```bash
    git checkout -b feature/nama-fitur
    ```

2. **Implementasi fitur:**

    - Tambahkan kode di direktori `src/` yang sesuai
    - Tambahkan tests di `tests/`
    - Update dokumentasi jika diperlukan

3. **Test dan format:**

    ```bash
    make test
    make format
    make lint
    ```

4. **Commit dan push:**

    ```bash
    git add .
    git commit -m "feat: deskripsi fitur"
    git push origin feature/nama-fitur
    ```

5. **Buat Pull Request**

## 📝 Commit Message Convention

Gunakan conventional commits:

-   `feat:` untuk fitur baru
-   `fix:` untuk bug fixes
-   `docs:` untuk dokumentasi
-   `style:` untuk formatting
-   `refactor:` untuk refactoring
-   `test:` untuk testing
-   `chore:` untuk maintenance

## 🐛 Bug Reports

Saat melaporkan bug, sertakan:

-   Deskripsi masalah
-   Steps to reproduce
-   Expected vs actual behavior
-   Environment info (OS, Python version, dll)
-   Error logs jika ada

## 💡 Feature Requests

Untuk request fitur baru:

-   Jelaskan use case
-   Berikan contoh implementasi jika memungkinkan
-   Diskusikan di Issues sebelum implementasi

## 📋 Code Review Guidelines

-   Code harus pass semua tests
-   Follow coding standards (Black, flake8)
-   Include appropriate tests
-   Update documentation jika diperlukan
-   Keep PR focused dan tidak terlalu besar

## 🔍 Debugging

### Common Issues

1. **Import errors:**

    - Pastikan `PYTHONPATH` include `src/`
    - Atau install package dalam development mode: `pip install -e .`

2. **Missing dependencies:**

    ```bash
    make install-dev
    ```

3. **Test failures:**
    ```bash
    pytest tests/ -v --tb=long
    ```

## 📞 Getting Help

-   Buka Issue untuk pertanyaan
-   Check existing Issues dan PRs
-   Review dokumentasi di `docs/`

## 🙏 Recognition

Semua kontributor akan diakui dalam CONTRIBUTORS.md file.
