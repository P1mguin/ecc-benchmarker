from os import urandom
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
from math import log

from ecc.math_utils.mod_inverse import modinv
from ecc.math_utils.mod_sqrt import modsqrt
from ecc.utils import int_length_in_byte


@dataclass
class Point:
    x: Optional[int]
    y: Optional[int]
    curve: "Curve"

    def is_at_infinity(self) -> bool:
        return self.x is None and self.y is None

    def __post_init__(self):
        if not self.is_at_infinity() and not self.curve.is_on_curve(self):
            raise ValueError("The point is not on the curve.")

    def __str__(self):
        if self.is_at_infinity():
            return f"Point(At infinity, Curve={str(self.curve)})"
        else:
            return f"Point(X={self.x}, Y={self.y}, Curve={str(self.curve)})"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.curve == other.curve and self.x == other.x and self.y == other.y

    def __neg__(self):
        return self.curve.neg_point(self)

    def __add__(self, other):
        return self.curve.add_point(self, other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        negative = - other
        return self.__add__(negative)

    def __mul__(self, scalar: int):
        return self.curve.mul_point(scalar, self)

    def __rmul__(self, scalar: int):
        return self.__mul__(scalar)


@dataclass
class Curve(ABC):
    name: str
    a: int
    b: int
    p: int
    n: int
    G_x: int
    G_y: int

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return (
            self.a == other.a and self.b == other.b and self.p == other.p and
            self.n == other.n and self.G_x == other.G_x and self.G_y == other.G_y
        )

    @property
    def G(self) -> Point:
        return Point(self.G_x, self.G_y, self)

    @property
    def INF(self) -> Point:
        return Point(None, None, self)

    def is_on_curve(self, P: Point) -> bool:
        if P.curve != self:
            return False
        return P.is_at_infinity() or self._is_on_curve(P)

    @abstractmethod
    def _is_on_curve(self, P: Point) -> bool:
        pass

    def add_point(self, P: Point, Q: Point) -> Point:
        if (not self.is_on_curve(P)) or (not self.is_on_curve(Q)):
            raise ValueError("The points are not on the curve.")
        if P.is_at_infinity():
            return Q
        elif Q.is_at_infinity():
            return P

        if P == -Q:
            return self.INF
        if P == Q:
            return self._double_point(P)

        return self._add_point(P, Q)

    @abstractmethod
    def _add_point(self, P: Point, Q: Point) -> Point:
        pass

    @abstractmethod
    def _double_point(self, P: Point) -> Point:
        pass

    def mul_point(self, d: int, P: Point) -> Point:
        """
        https://en.wikipedia.org/wiki/Elliptic_curve_point_multiplication
        """
        if not self.is_on_curve(P):
            raise ValueError("The point is not on the curve.")
        if P.is_at_infinity():
            return self.INF
        if d == 0:
            return self.INF

        res = self.INF
        is_negative_scalar = d < 0
        d = -d if is_negative_scalar else d
        tmp = P
        while d:
            if d & 0x1 == 1:
                res = self.add_point(res, tmp)
            tmp = self.add_point(tmp, tmp)
            d >>= 1
        if is_negative_scalar:
            return -res
        else:
            return res

    def neg_point(self, P: Point) -> Point:
        if not self.is_on_curve(P):
            raise ValueError("The point is not on the curve.")
        if P.is_at_infinity():
            return self.INF

        return self._neg_point(P)

    def get_maximum_size(self) -> int:
        return int(log(self.n, 2) // 8)

    def fits_size(self, size) -> bool:
        return size <= self.get_maximum_size()

    @abstractmethod
    def _neg_point(self, P: Point) -> Point:
        pass

    @abstractmethod
    def compute_y(self, x: int) -> int:
        pass

    def encode_point(self, plaintext: bytes) -> Point:
        plaintext = len(plaintext).to_bytes(1, byteorder="big") + plaintext

        while True:
            x = int.from_bytes(plaintext, "big")
            y = self.compute_y(x)
            if y:
                return Point(x, y, self)
            plaintext += urandom(1)

    def decode_point(self, M: Point) -> bytes:
        byte_len = int_length_in_byte(M.x)
        plaintext_len = (M.x >> ((byte_len - 1) * 8)) & 0xff
        plaintext = ((M.x >> ((byte_len - plaintext_len - 1) * 8))
                     & (int.from_bytes(b"\xff" * plaintext_len, "big")))
        return plaintext.to_bytes(plaintext_len, byteorder="big")


class ShortWeierstrassCurve(Curve):
    """
    y^2 = x^3 + a*x + b
    https://en.wikipedia.org/wiki/Elliptic_curve
    """

    def _is_on_curve(self, P: Point) -> bool:
        left = P.y * P.y
        right = (P.x * P.x * P.x) + (self.a * P.x) + self.b
        return (left - right) % self.p == 0

    def _add_point(self, P: Point, Q: Point) -> Point:
        # s = (yP - yQ) / (xP - xQ)
        # xR = s^2 - xP - xQ
        # yR = yP + s * (xR - xP)
        delta_x = P.x - Q.x
        delta_y = P.y - Q.y
        s = delta_y * modinv(delta_x, self.p)
        res_x = (s * s - P.x - Q.x) % self.p
        res_y = (P.y + s * (res_x - P.x)) % self.p
        return - Point(res_x, res_y, self)

    def _double_point(self, P: Point) -> Point:
        # s = (3 * xP^2 + a) / (2 * yP)
        # xR = s^2 - 2 * xP
        # yR = yP + s * (xR - xP)
        s = (3 * P.x * P.x + self.a) * modinv(2 * P.y, self.p)
        res_x = (s * s - 2 * P.x) % self.p
        res_y = (P.y + s * (res_x - P.x)) % self.p
        return - Point(res_x, res_y, self)

    def _neg_point(self, P: Point) -> Point:
        return Point(P.x, -P.y % self.p, self)

    def compute_y(self, x) -> int:
        right = (x * x * x + self.a * x + self.b) % self.p
        y = modsqrt(right, self.p)
        return y


class ShortWeierstrassBinaryCurve(Curve):
    """
    y^2 + x*y = x^3 + a*x^2 + b
    https://en.wikipedia.org/wiki/Elliptic_curve
    """

    def _is_on_curve(self, P: Point) -> bool:
        left = P.y * P.y + P.x * P.y
        right = (P.x * P.x * P.x) + (self.a * P.x * P.x) + self.b
        return (left - right) % self.p == 0

    def _add_point(self, P: Point, Q: Point) -> Point:
        # s = (yP - yQ) / (xP - xQ)
        # xR = s*s + s - xP - xQ - a
        # yR = s(xQ - xR) - xR - yQ
        s = (P.y - Q.y) / (P.x - Q.x)
        res_x = s * s + s - P.x - Q.x - self.a
        res_y = s * (Q.x - res_x) - res_x - Q.y
        return - Point(int(res_x % self.p), int(res_y % self.p), self)

    def _double_point(self, P: Point) -> Point:
        # s = (3 * Px * Px + 2 * a * Px - Py) / (2 * Py + Px)
        # xR = s * s + s - 2 * Px - a
        # yR = s * (Px - Rx) - Rx - Py
        s = (3 * P.x * P.x + 2 * self.a * P.x - P.y) / (2 * P.y + P.x)
        res_x = s * s + s - 2 * P.x - self.a
        res_y = s * (P.x - res_x) - res_x - P.y
        return - Point(int(res_x % self.p), int(res_y % self.p), self)

    def _neg_point(self, P: Point) -> Point:
        return Point(P.x, -(P.y + P.x) % self.p, self)
        # return Point(P.x, -P.y % self.p, self)

    def compute_y(self, x) -> int:
        return self._compute_up_y(x)

    def _compute_up_y(self, x) -> int:
        # y = (sqrt(4*a*x^2 + 4*b + 4*x^3 + x^2) - x)/2
        y = (pow(4 * self.a * x * x + 4 * self.b + 4 * x * x * x + x * x, 1 / 2) - x) / 2
        return y % self.p

    def _compute_down_y(self, x) -> int:
        # y = (-sqrt(4*a*x^2 + 4*b + 4*x^3 + x^2) - x)/2
        y = (-pow(4 * self.a * x * x + 4 * self.b + 4 * x * x * x + x * x, 1 / 2) - x) / 2
        return y % self.p


class MontgomeryCurve(Curve):
    """
    by^2 = x^3 + ax^2 + x
    https://en.wikipedia.org/wiki/Montgomery_curve
    """

    def _is_on_curve(self, P: Point) -> bool:
        left = self.b * P.y * P.y
        right = (P.x * P.x * P.x) + (self.a * P.x * P.x) + P.x
        return (left - right) % self.p == 0

    def _add_point(self, P: Point, Q: Point) -> Point:
        # s = (yP - yQ) / (xP - xQ)
        # xR = b * s^2 - a - xP - xQ
        # yR = yP + s * (xR - xP)
        delta_x = P.x - Q.x
        delta_y = P.y - Q.y
        s = delta_y * modinv(delta_x, self.p)
        res_x = (self.b * s * s - self.a - P.x - Q.x) % self.p
        res_y = (P.y + s * (res_x - P.x)) % self.p
        return - Point(res_x, res_y, self)

    def _double_point(self, P: Point) -> Point:
        # s = (3 * xP^2 + 2 * a * xP + 1) / (2 * b * yP)
        # xR = b * s^2 - a - 2 * xP
        # yR = yP + s * (xR - xP)
        up = 3 * P.x * P.x + 2 * self.a * P.x + 1
        down = 2 * self.b * P.y
        s = up * modinv(down, self.p)
        res_x = (self.b * s * s - self.a - 2 * P.x) % self.p
        res_y = (P.y + s * (res_x - P.x)) % self.p
        return - Point(res_x, res_y, self)

    def _neg_point(self, P: Point) -> Point:
        return Point(P.x, -P.y % self.p, self)

    def compute_y(self, x: int) -> int:
        right = (x * x * x + self.a * x * x + x) % self.p
        inv_b = modinv(self.b, self.p)
        right = (right * inv_b) % self.p
        y = modsqrt(right, self.p)
        return y


class TwistedEdwardsCurve(Curve):
    """
    ax^2 + y^2 = 1 + bx^2y^2
    https://en.wikipedia.org/wiki/Twisted_Edwards_curve
    """
    def _is_on_curve(self, P: Point) -> bool:
        left = self.a * P.x * P.x + P.y * P.y
        right = 1 + self.b * P.x * P.x * P.y * P.y
        return (left - right) % self.p == 0

    def _add_point(self, P: Point, Q: Point) -> Point:
        # xR = (xP * yQ + yP * xQ) / (1 + b * xP * xQ * yP * yQ)
        up_x = P.x * Q.y + P.y * Q.x
        down_x = 1 + self.b * P.x * Q.x * P.y * Q.y
        res_x = (up_x * modinv(down_x, self.p)) % self.p
        # yR = (yP * yQ - a * xP * xQ) / (1 - b * xP * xQ * yP * yQ)
        up_y = P.y * Q.y - self.a * P.x * Q.x
        down_y = 1 - self.b * P.x * Q.x * P.y * Q.y
        res_y = (up_y * modinv(down_y, self.p)) % self.p
        return Point(res_x, res_y, self)

    def _double_point(self, P: Point) -> Point:
        # xR = (2 * xP * yP) / (a * xP^2 + yP^2)
        up_x = 2 * P.x * P.y
        down_x = self.a * P.x * P.x + P.y * P.y
        res_x = (up_x * modinv(down_x, self.p)) % self.p
        # yR = (yP^2 - a * xP * xP) / (2 - a * xP^2 - yP^2)
        up_y = P.y * P.y - self.a * P.x * P.x
        down_y = 2 - self.a * P.x * P.x - P.y * P.y
        res_y = (up_y * modinv(down_y, self.p)) % self.p
        return Point(res_x, res_y, self)

    def _neg_point(self, P: Point) -> Point:
        return Point(-P.x % self.p, P.y, self)

    def compute_y(self, x: int) -> int:
        # (bx^2 - 1) * y^2 = ax^2 - 1
        right = self.a * x * x - 1
        left_scale = (self.b * x * x - 1) % self.p
        inv_scale = modinv(left_scale, self.p)
        right = (right * inv_scale) % self.p
        y = modsqrt(right, self.p)
        return y


P192 = ShortWeierstrassCurve(
    name="P192",
    a=-3,
    b=2455155546008943817740293915197451784769108058161191238065,
    p=0xfffffffffffffffffffffffffffffffeffffffffffffffff,
    n=0xffffffffffffffffffffffff99def836146bc9b1b4d22831,
    G_x=0x188da80eb03090f67cbf20eb43a18800f4ff0afd82ff1012,
    G_y=0x07192b95ffc8da78631011ed6b24cdd573f977a11e794811
)

P224 = ShortWeierstrassCurve(
    name="P224",
    a=-3,
    b=18958286285566608000408668544493926415504680968679321075787234672564,
    p=0xffffffffffffffffffffffffffffffff000000000000000000000001,
    n=0xffffffffffffffffffffffffffff16a2e0b8f03e13dd29455c5c2a3d,
    G_x=0xb70e0cbd6bb4bf7f321390b94a03c1d356c21122343280d6115c1d21,
    G_y=0xbd376388b5f723fb4c22dfe6cd4375a05a07476444d5819985007e34
)

P256 = ShortWeierstrassCurve(
    name="P256",
    a=-3,
    b=41058363725152142129326129780047268409114441015993725554835256314039467401291,
    p=0xffffffff00000001000000000000000000000000ffffffffffffffffffffffff,
    n=0xffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551,
    G_x=0x6b17d1f2e12c4247f8bce6e563a440f277037d812deb33a0f4a13945d898c296,
    G_y=0x4fe342e2fe1a7f9b8ee7eb4a7c0f9e162bce33576b315ececbb6406837bf51f5
)

P384 = ShortWeierstrassCurve(
    name="P384",
    a=-3,
    b=27580193559959705877849011840389048093056905856361568521428707301988689241309860865136260764883745107765439761230575,
    p=0xfffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffeffffffff0000000000000000ffffffff,
    n=0xffffffffffffffffffffffffffffffffffffffffffffffffc7634d81f4372ddf581a0db248b0a77aecec196accc52973,
    G_x=0xaa87ca22be8b05378eb1c71ef320ad746e1d3b628ba79b9859f741e082542a385502f25dbf55296c3a545e3872760ab7,
    G_y=0x3617de4a96262c6f5d9e98bf9292dc29f8f41dbd289a147ce9da3113b5f0b8c00a60b1ce1d7e819d7a431d7c90ea0e5f
)

P521 = ShortWeierstrassCurve(
    name="P521",
    a=-3,
    b=1093849038073734274511112390766805569936207598951683748994586394495953116150735016013708737573759623248592132296706313309438452531591012912142327488478985984,
    p=0x1ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff,
    n=6864797660130609714981900799081393217269435300143305409394463459185543183397655394245057746333217197532963996371363321113864768612440380340372808892707005449,
    G_x=2661740802050217063228768716723360960729859168756973147706671368418802944996427808491545080627771902352094241225065558662157113545570916814161637315895999846,
    G_y=3757180025770020463545507224491183603594455134769762486694567779615544477440556316691234405012945539562144444537289428522585666729196580810124344277578376784
)

W25519 = ShortWeierstrassCurve(
    name="W-25519",
    a=19298681539552699237261830834781317975544997444273427339909597334573241639236,
    b=55751746669818908907645289078257140818241103727901012315294400837956729358436,
    p=0x7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffed,
    n=7237005577332262213973186563042994240857116359379907606001950938285454250989,
    G_x=19298681539552699237261830834781317975544997444273427339909597334652188435546,
    G_y=43114425171068552920764898935933967039370386198203806730763910166200978582548
)

W448 = ShortWeierstrassCurve(
    name="W-448",
    a=0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa9fffffffffffffffffffffffffffffffffffffffffffffffe1a76d41f,
    b=0x5ed097b425ed097b425ed097b425ed097b425ed097b425ed097b425e71c71c71c71c71c71c71c71c71c71c71c71c71c71c72c87b7cc69f70,
    p=0xfffffffffffffffffffffffffffffffffffffffffffffffffffffffeffffffffffffffffffffffffffffffffffffffffffffffffffffffff,
    n=181709681073901722637330951972001133588410340171829515070372549795146003961539585716195755291692375963310293709091662304773755859649779,
    G_x=0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa0000000000000000000000000000000000000000000000000000cb91,
    G_y=0x7d235d1295f5b1f66c98ab6e58326fcecbae5d34f55545d060f75dc28df3f6edb8027e2346430d211312c4b150677af76fd7223d457b5b1a
)

Curve25519 = MontgomeryCurve(
    name="Curve25519",
    a=486662,
    b=1,
    p=0x7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffed,
    n=0x1000000000000000000000000000000014def9dea2f79cd65812631a5cf5d3ed,
    G_x=0x9,
    G_y=0x20ae19a1b8a086b4e01edd2c7748d14c923d4d7e6d7c61b229e9c5a27eced3d9
)

Curve448 = MontgomeryCurve(
    name="Curve448",
    a=156326,
    b=1,
    p=0xfffffffffffffffffffffffffffffffffffffffffffffffffffffffeffffffffffffffffffffffffffffffffffffffffffffffffffffffff,
    n=181709681073901722637330951972001133588410340171829515070372549795146003961539585716195755291692375963310293709091662304773755859649779,
    G_x=0x5,
    G_y=0x7d235d1295f5b1f66c98ab6e58326fcecbae5d34f55545d060f75dc28df3f6edb8027e2346430d211312c4b150677af76fd7223d457b5b1a
)

# Brainpool curves
BrainpoolP160r1 = ShortWeierstrassCurve(
    name="BrainpoolP160r1",
    a=0x340E7BE2A280EB74E2BE61BADA745D97E8F7C300,
    b=0x1E589A8595423412134FAA2DBDEC95C8D8675E58,
    p=0xe95e4a5f737059dc60dfc7ad95b3d8139515620f,
    n=0xE95E4A5F737059DC60DF5991D45029409E60FC09,
    G_x=0xBED5AF16EA3F6A4F62938C4631EB5AF7BDBCDBC3,
    G_y=0x1667CB477A1A8EC338F94741669C976316DA6321
)

BrainpoolP160t1 = ShortWeierstrassCurve(
    name="BrainpoolP160t1",
    a=0xe95e4a5f737059dc60dfc7ad95b3d8139515620c,
    b=0x7a556b6dae535b7b51ed2c4d7daa7a0b5c55f380,
    p=0xe95e4a5f737059dc60dfc7ad95b3d8139515620f,
    n=0xe95e4a5f737059dc60df5991d45029409e60fc09,
    G_x=0xb199b13b9b34efc1397e64baeb05acc265ff2378,
    G_y=0xadd6718b7c7c1961f0991b842443772152c9e0ad,
)

BrainpoolP192r1 = ShortWeierstrassCurve(
    name="BrainpoolP192r1",
    a=0x6A91174076B1E0E19C39C031FE8685C1CAE040E5C69A28EF,
    b=0x469A28EF7C28CCA3DC721D044F4496BCCA7EF4146FBF25C9,
    p=0xC302F41D932A36CDA7A3463093D18DB78FCE476DE1A86297,
    n=0xC302F41D932A36CDA7A3462F9E9E916B5BE8F1029AC4ACC1,
    G_x=0xC0A0647EAAB6A48753B033C56CB0F0900A2F5C4853375FD6,
    G_y=0x14B690866ABD5BB88B5F4828C1490002E6773FA2FA299B8F
)

BrainpoolP192t1 = ShortWeierstrassCurve(
    name="BrainpoolP192t1",
    a=0xc302f41d932a36cda7a3463093d18db78fce476de1a86294,
    b=0x13d56ffaec78681e68f9deb43b35bec2fb68542e27897b79,
    p=0xc302f41d932a36cda7a3463093d18db78fce476de1a86297,
    n=0xc302f41d932a36cda7a3462f9e9e916b5be8f1029ac4acc1,
    G_x=0x3ae9e58c82f63c30282e1fe7bbf43fa72c446af6f4618129,
    G_y=0x97e2c5667c2223a902ab5ca449d0084b7e5b3de7ccc01c9
)

BrainpoolP224r1 = ShortWeierstrassCurve(
    name="BrainpoolP224r1",
    a=0x68a5e62ca9ce6c1c299803a6c1530b514e182ad8b0042a59cad29f43,
    b=0x2580f63ccfe44138870713b1a92369e33e2135d266dbb372386c400b,
    p=0xd7c134aa264366862a18302575d1d787b09f075797da89f57ec8c0ff,
    n=0xd7c134aa264366862a18302575d0fb98d116bc4b6ddebca3a5a7939f,
    G_x=0xd9029ad2c7e5cf4340823b2a87dc68c9e4ce3174c1e6efdee12c07d,
    G_y=0x58aa56f772c0726f24c6b89e4ecdac24354b9e99caa3f6d3761402cd
)

BrainpoolP224t1 = ShortWeierstrassCurve(
    name="BrainpoolP224t1",
    a=0xd7c134aa264366862a18302575d1d787b09f075797da89f57ec8c0fc,
    b=0x4b337d934104cd7bef271bf60ced1ed20da14c08b3bb64f18a60888d,
    p=0xd7c134aa264366862a18302575d1d787b09f075797da89f57ec8c0ff,
    n=0xd7c134aa264366862a18302575d0fb98d116bc4b6ddebca3a5a7939f,
    G_x=0x6ab1e344ce25ff3896424e7ffe14762ecb49f8928ac0c76029b4d580,
    G_y=0x374e9f5143e568cd23f3f4d7c0d4b1e41c8cc0d1c6abd5f1a46db4c
)

BrainpoolP256r1 = ShortWeierstrassCurve(
    name="BrainpoolP256r1",
    a=0x7d5a0975fc2c3057eef67530417affe7fb8055c126dc5c6ce94a4b44f330b5d9,
    b=0x26dc5c6ce94a4b44f330b5d9bbd77cbf958416295cf7e1ce6bccdc18ff8c07b6,
    p=0xa9fb57dba1eea9bc3e660a909d838d726e3bf623d52620282013481d1f6e5377,
    n=0xa9fb57dba1eea9bc3e660a909d838d718c397aa3b561a6f7901e0e82974856a7,
    G_x=0x8bd2aeb9cb7e57cb2c4b482ffc81b7afb9de27e1e3bd23c23a4453bd9ace3262,
    G_y=0x547ef835c3dac4fd97f8461a14611dc9c27745132ded8e545c1d54c72f046997,
)

BrainpoolP256t1 = ShortWeierstrassCurve(
    name="BrainpoolP256t1",
    a=0xa9fb57dba1eea9bc3e660a909d838d726e3bf623d52620282013481d1f6e5374,
    b=0x662c61c430d84ea4fe66a7733d0b76b7bf93ebc4af2f49256ae58101fee92b04,
    p=0xa9fb57dba1eea9bc3e660a909d838d726e3bf623d52620282013481d1f6e5377,
    n=0xa9fb57dba1eea9bc3e660a909d838d718c397aa3b561a6f7901e0e82974856a7,
    G_x=0xa3e8eb3cc1cfe7b7732213b23a656149afa142c47aafbc2b79a191562e1305f4,
    G_y=0x2d996c823439c56d7f7b22e14644417e69bcb6de39d027001dabe8f35b25c9be
)

BrainpoolP320r1 = ShortWeierstrassCurve(
    name="BrainpoolP320r1",
    a=0x3ee30b568fbab0f883ccebd46d3f3bb8a2a73513f5eb79da66190eb085ffa9f492f375a97d860eb4,
    b=0x520883949dfdbc42d3ad198640688a6fe13f41349554b49acc31dccd884539816f5eb4ac8fb1f1a6,
    p=0xd35e472036bc4fb7e13c785ed201e065f98fcfa6f6f40def4f92b9ec7893ec28fcd412b1f1b32e27,
    n=0xd35e472036bc4fb7e13c785ed201e065f98fcfa5b68f12a32d482ec7ee8658e98691555b44c59311,
    G_x=0x43bd7e9afb53d8b85289bcc48ee5bfe6f20137d10a087eb6e7871e2a10a599c710af8d0d39e20611,
    G_y=0x14fdd05545ec1cc8ab4093247f77275e0743ffed117182eaa9c77877aaac6ac7d35245d1692e8ee1
)

BrainpoolP320t1 = ShortWeierstrassCurve(
    name="BrainpoolP320t1",
    a=0xd35e472036bc4fb7e13c785ed201e065f98fcfa6f6f40def4f92b9ec7893ec28fcd412b1f1b32e24,
    b=0xa7f561e038eb1ed560b3d147db782013064c19f27ed27c6780aaf77fb8a547ceb5b4fef422340353,
    p=0xd35e472036bc4fb7e13c785ed201e065f98fcfa6f6f40def4f92b9ec7893ec28fcd412b1f1b32e27,
    n=0xd35e472036bc4fb7e13c785ed201e065f98fcfa5b68f12a32d482ec7ee8658e98691555b44c59311,
    G_x=0x925be9fb01afc6fb4d3e7d4990010f813408ab106c4f09cb7ee07868cc136fff3357f624a21bed52,
    G_y=0x63ba3a7a27483ebf6671dbef7abb30ebee084e58a0b077ad42a5a0989d1ee71b1b9bc0455fb0d2c3
)

BrainpoolP384r1 = ShortWeierstrassCurve(
    name="BrainpoolP384r1",
    a=0x7bc382c63d8c150c3c72080ace05afa0c2bea28e4fb22787139165efba91f90f8aa5814a503ad4eb04a8c7dd22ce2826,
    b=0x4a8c7dd22ce28268b39b55416f0447c2fb77de107dcd2a62e880ea53eeb62d57cb4390295dbc9943ab78696fa504c11,
    p=0x8cb91e82a3386d280f5d6f7e50e641df152f7109ed5456b412b1da197fb71123acd3a729901d1a71874700133107ec53,
    n=0x8cb91e82a3386d280f5d6f7e50e641df152f7109ed5456b31f166e6cac0425a7cf3ab6af6b7fc3103b883202e9046565,
    G_x=0x1d1c64f068cf45ffa2a63a81b7c13f6b8847a3e77ef14fe3db7fcafe0cbd10e8e826e03436d646aaef87b2e247d4af1e,
    G_y=0x8abe1d7520f9c2a45cb1eb8e95cfd55262b70b29feec5864e19c054ff99129280e4646217791811142820341263c5315
)

BrainpoolP384t1 = ShortWeierstrassCurve(
    name="BrainpoolP384t1",
    a=0x8cb91e82a3386d280f5d6f7e50e641df152f7109ed5456b412b1da197fb71123acd3a729901d1a71874700133107ec50,
    b=0x7f519eada7bda81bd826dba647910f8c4b9346ed8ccdc64e4b1abd11756dce1d2074aa263b88805ced70355a33b471ee,
    p=0x8cb91e82a3386d280f5d6f7e50e641df152f7109ed5456b412b1da197fb71123acd3a729901d1a71874700133107ec53,
    n=0x8cb91e82a3386d280f5d6f7e50e641df152f7109ed5456b31f166e6cac0425a7cf3ab6af6b7fc3103b883202e9046565,
    G_x=0x18de98b02db9a306f2afcd7235f72a819b80ab12ebd653172476fecd462aabffc4ff191b946a5f54d8d0aa2f418808cc,
    G_y=0x25ab056962d30651a114afd2755ad336747f93475b7a1fca3b88f2b6a208ccfe469408584dc2b2912675bf5b9e582928
)

BrainpoolP512r1 = ShortWeierstrassCurve(
    name="BrainpoolP512r1",
    a=0x7830a3318b603b89e2327145ac234cc594cbdd8d3df91610a83441caea9863bc2ded5d5aa8253aa10a2ef1c98b9ac8b57f1117a72bf2c7b9e7c1ac4d77fc94ca,
    b=0x3df91610a83441caea9863bc2ded5d5aa8253aa10a2ef1c98b9ac8b57f1117a72bf2c7b9e7c1ac4d77fc94cadc083e67984050b75ebae5dd2809bd638016f723,
    p=0xaadd9db8dbe9c48b3fd4e6ae33c9fc07cb308db3b3c9d20ed6639cca703308717d4d9b009bc66842aecda12ae6a380e62881ff2f2d82c68528aa6056583a48f3,
    n=0xaadd9db8dbe9c48b3fd4e6ae33c9fc07cb308db3b3c9d20ed6639cca70330870553e5c414ca92619418661197fac10471db1d381085ddaddb58796829ca90069,
    G_x=0x81aee4bdd82ed9645a21322e9c4c6a9385ed9f70b5d916c1b43b62eef4d0098eff3b1f78e2d0d48d50d1687b93b97d5f7c6d5047406a5e688b352209bcb9f822,
    G_y=0x7dde385d566332ecc0eabfa9cf7822fdf209f70024a57b1aa000c55b881f8111b2dcde494a5f485e5bca4bd88a2763aed1ca2b2fa8f0540678cd1e0f3ad80892
)

BrainpoolP512t1 = ShortWeierstrassCurve(
    name="Brainpool512t1",
    a=0xaadd9db8dbe9c48b3fd4e6ae33c9fc07cb308db3b3c9d20ed6639cca703308717d4d9b009bc66842aecda12ae6a380e62881ff2f2d82c68528aa6056583a48f0,
    b=0x7cbbbcf9441cfab76e1890e46884eae321f70c0bcb4981527897504bec3e36a62bcdfa2304976540f6450085f2dae145c22553b465763689180ea2571867423e,
    p=0xaadd9db8dbe9c48b3fd4e6ae33c9fc07cb308db3b3c9d20ed6639cca703308717d4d9b009bc66842aecda12ae6a380e62881ff2f2d82c68528aa6056583a48f3,
    n=0xaadd9db8dbe9c48b3fd4e6ae33c9fc07cb308db3b3c9d20ed6639cca70330870553e5c414ca92619418661197fac10471db1d381085ddaddb58796829ca90069,
    G_x=0x640ece5c12788717b9c1ba06cbc2a6feba85842458c56dde9db1758d39c0313d82ba51735cdb3ea499aa77a7d6943a64f7a3f25fe26f06b51baa2696fa9035da,
    G_y=0x5b534bd595f5af0fa2c892376c84ace1bb4e3019b71634c01131159cae03cee9d9932184beef216bd71df2dadf86a627306ecff96dbb8bace198b61e00f8b332
)

#Edwards curves
Ed25519 = TwistedEdwardsCurve(
    name="Edwards25519",
    a=0x7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffec,
    b=0x52036cee2b6ffe738cc740797779e89800700a4d4141d8ab75eb4dca135978a3,
    p=0x7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffed,
    n=0x1000000000000000000000000000000014def9dea2f79cd65812631a5cf5d3ed,
    G_x=0x216936D3CD6E53FEC0A4E231FDD6DC5C692CC7609525A7B2C9562D608F25D51A,
    G_y=0x6666666666666666666666666666666666666666666666666666666666666658
)

Ed448 = TwistedEdwardsCurve(
    name="Edwards448",
    a=0x01,
    b=0xfffffffffffffffffffffffffffffffffffffffffffffffffffffffeffffffffffffffffffffffffffffffffffffffffffffffffffff6756,
    p=0xfffffffffffffffffffffffffffffffffffffffffffffffffffffffeffffffffffffffffffffffffffffffffffffffffffffffffffffffff,
    n=0x3fffffffffffffffffffffffffffffffffffffffffffffffffffffff7cca23e9c44edb49aed63690216cc2728dc58f552378c292ab5844f3,
    G_x=0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa955555555555555555555555555555555555555555555555555555555,
    G_y=0xae05e9634ad7048db359d6205086c2b0036ed7a035884dd7b7e36d728ad8c4b80d6565833a2a3098bbbcb2bed1cda06bdaeafbcdea9386ed
)

ShortWeierstrassCurves = [P192, P224, P256, P384, P521, W25519, W448]
MontgomeryCurves = [Curve25519, Curve448]
BrainpoolCurvesR1 = [BrainpoolP160r1, BrainpoolP192r1, BrainpoolP224r1, BrainpoolP256r1, BrainpoolP320r1, BrainpoolP384r1, BrainpoolP512r1]
BrainpoolCurvesT1 = [BrainpoolP160t1, BrainpoolP192t1, BrainpoolP224t1, BrainpoolP256t1, BrainpoolP320t1, BrainpoolP384t1, BrainpoolP512t1]
EdwardsCurves = [Ed25519, Ed448]
