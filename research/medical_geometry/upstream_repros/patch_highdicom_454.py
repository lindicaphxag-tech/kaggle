"""Apply a two-site, pinned minimal patch to a local highdicom #454 checkout.

Usage: python patch_highdicom_454.py ../upstream-highdicom/src/highdicom/volume.py
Aborts when the audited source anchors do not match; no remote files edited.
"""

from pathlib import Path
import sys

UID_META_KEY = 'highdicom_frame_of_reference_uid'

TO_OLD = '''        meta[MetaKeys.ORIGINAL_CHANNEL_DIM] = (
            -1 if self.ndim > 3 else float("nan")
        )
'''
TO_NEW = TO_OLD + '''        if self.frame_of_reference_uid is not None:
            meta["highdicom_frame_of_reference_uid"] = str(self.frame_of_reference_uid)
'''

FROM_OLD = '''        array = metatensor.cpu().detach().numpy()

        return cls(
'''
FROM_NEW = '''        array = metatensor.cpu().detach().numpy()

        if frame_of_reference_uid is None:
            frame_of_reference_uid = metatensor.meta.get(
                "highdicom_frame_of_reference_uid"
            )

        return cls(
'''


def apply(path: Path) -> None:
    text = path.read_text(encoding='utf-8')
    for name, old, new in (
        ('to_monai', TO_OLD, TO_NEW),
        ('from_monai', FROM_OLD, FROM_NEW),
    ):
        if text.count(old) != 1:
            raise RuntimeError(
                f'{name} pinned source anchor missing/ambiguous: {text.count(old)}'
            )
        text = text.replace(old, new, 1)
    path.write_text(text, encoding='utf-8')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('usage: patch_highdicom_454.py <path/to/highdicom/volume.py>')
    apply(Path(sys.argv[1]))
    print('Applied pinned highdicom #454 FrameOfReferenceUID patch')
