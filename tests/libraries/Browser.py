class Browser:
    def new_browser(self, *, browser: str | None = None, headless: bool = True, **options):
        return {'browser': browser, 'headless': headless, **options}
