"""Import of Substance Designer packages (.sbs), one way: Designer's File > Open of a .sbs gives a Trafilatura 3D package
(saved as .trafilatura3d). See importer for what is converted and how what is not is kept visible."""

from trafilatura3d.sbs.importer import Report, import_sbs

__all__ = ["Report", "import_sbs"]
