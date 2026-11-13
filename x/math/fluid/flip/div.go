package flip

// The step divides about 6,300 times (32×24 cells, 700 particles): the
// FLIP blend's weighted means, the transfer's normalisation and the
// separation's pushes. The badge's core (Cortex-M0+) has no divide
// instruction, and TinyGo calls a software routine that shifts and
// subtracts per quotient bit, even for a constant divisor. So under TinyGo
// div is sdiv, exact and divide-free (div_tinygo.go); elsewhere it is Go's
// / (div_gc.go), which a host's divider does faster. Both give the same
// quotients, so the liquid is the same on the badge and the host.

// recip16[d] = ⌊2¹⁶/d⌋ for 2 ≤ d ≤ 256: udiv's reciprocals. A literal, so
// TinyGo keeps it in flash (built in a loop it was RAM, a heap object, and
// 37 KB of code storing it).
var recip16 = [257]uint16{
	0, 0, 32768, 21845, 16384, 13107, 10922, 9362, 8192, 7281, 6553, 5957, 5461, 5041, 4681, 4369,
	4096, 3855, 3640, 3449, 3276, 3120, 2978, 2849, 2730, 2621, 2520, 2427, 2340, 2259, 2184, 2114,
	2048, 1985, 1927, 1872, 1820, 1771, 1724, 1680, 1638, 1598, 1560, 1524, 1489, 1456, 1424, 1394,
	1365, 1337, 1310, 1285, 1260, 1236, 1213, 1191, 1170, 1149, 1129, 1110, 1092, 1074, 1057, 1040,
	1024, 1008, 992, 978, 963, 949, 936, 923, 910, 897, 885, 873, 862, 851, 840, 829,
	819, 809, 799, 789, 780, 771, 762, 753, 744, 736, 728, 720, 712, 704, 697, 689,
	682, 675, 668, 661, 655, 648, 642, 636, 630, 624, 618, 612, 606, 601, 595, 590,
	585, 579, 574, 569, 564, 560, 555, 550, 546, 541, 537, 532, 528, 524, 520, 516,
	512, 508, 504, 500, 496, 492, 489, 485, 481, 478, 474, 471, 468, 464, 461, 458,
	455, 451, 448, 445, 442, 439, 436, 434, 431, 428, 425, 422, 420, 417, 414, 412,
	409, 407, 404, 402, 399, 397, 394, 392, 390, 387, 385, 383, 381, 378, 376, 374,
	372, 370, 368, 366, 364, 362, 360, 358, 356, 354, 352, 350, 348, 346, 344, 343,
	341, 339, 337, 336, 334, 332, 330, 329, 327, 326, 324, 322, 321, 319, 318, 316,
	315, 313, 312, 310, 309, 307, 306, 304, 303, 302, 300, 299, 297, 296, 295, 293,
	292, 291, 289, 288, 287, 286, 284, 283, 282, 281, 280, 278, 277, 276, 275, 274,
	273, 271, 270, 269, 268, 267, 266, 265, 264, 263, 262, 261, 260, 259, 258, 257,
	256,
}

// divMax bounds the divisors div takes.
const divMax = 1024

// udiv is ⌊n/d⌋ for n ≤ 2³¹ and 2 ≤ d < divMax, exactly and without a
// divide: an estimate from a reciprocal no larger than 2¹⁶/d (above 256,
// that of ⌈d/2⌉ or ⌈d/4⌉ shifted down, so every part rounds down and never
// overshoots), then remainder steps until exact; each product stays under
// 2³².
func udiv(n, d uint32) uint32 {
	var r uint32
	switch {
	case d <= 256:
		r = uint32(recip16[d])
	case d < 512:
		r = uint32(recip16[(d+1)>>1]) >> 1
	default:
		r = uint32(recip16[(d+3)>>2]) >> 2
	}
	q := n>>16*r + (n&0xffff)*r>>16
	n -= q * d
	for n >= d {
		t := max(n*r>>16, 1)
		q += t
		n -= t * d
	}
	return q
}

// sdiv is n/d truncated as Go's / truncates it, for 1 ≤ d < divMax,
// without a divide.
func sdiv(n, d int32) int32 {
	switch {
	case d == 1:
		return n
	case n < 0:
		return -int32(udiv(uint32(-n), uint32(d)))
	}
	return int32(udiv(uint32(n), uint32(d)))
}
