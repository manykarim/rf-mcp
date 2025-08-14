class VarArgLib:
    def Concat(self, *parts):
        return ",".join(parts)

    def UseKwArgs(self, **kwargs):
        return ";".join(f"{k}={v}" for k, v in sorted(kwargs.items()))

    def Mix(self, base, *parts, **options):
        text = base + ''.join(parts)
        if options:
            text += ';' + ';'.join(f"{k}={v}" for k, v in sorted(options.items()))
        return text
