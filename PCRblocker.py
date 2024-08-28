import math
import copy
import numpy as np
from Bio import SeqUtils, Seq
from Bio.SeqUtils import MeltingTemp

# Allawi and SantaLucia (1997), Biochemistry 36: 10581-10594
# Richard Owczarzy, Yong You, Christopher L. Groth, and Andrey V. Tataurov, Biochemistry, 50, 43, 9352-9367 2011.
# Confirmed correct on Mar 11, 2024
LNA_NN = {
    "init": (0, 0), "init_A/T": (2.3, 4.1), "init_G/C": (0.1, -2.8),
    "init_oneG/C": (0, 0), "init_allA/T": (0, 0), "init_5T/A": (0, 0),
    "sym": (0, -1.4),
    "AA/TT": (-7.9, -22.2), "AT/TA": (-7.2, -20.4), "TA/AT": (-7.2, -21.3),
    "CA/GT": (-8.5, -22.7), "GT/CA": (-8.4, -22.4), "CT/GA": (-7.8, -21.0),
    "GA/CT": (-8.2, -22.2), "CG/GC": (-10.6, -27.2), "GC/CG": (-9.8, -24.4),
    "GG/CC": (-8.0, -19.9),
    "aa/TT": (-9.991, -27.175), "ac/TG": (-11.389, -28.963), "ag/TC": (-12.793, -31.607),
    "at/TA": (-14.703, -40.750), "ca/GT": (-14.177, -35.498), "cc/GG": (-15.399, -36.375),
    "cg/GC": (-14.558, -35.239), "ct/GA": (-15.737, -41.218), "ga/CT": (-13.959, -35.097),
    "gc/CG": (-16.109, -40.738), "gg/CC": (-13.022, -29.673), "gt/CA": (-17.361, -45.858),
    "ta/AT": (-10.318, -26.108), "tc/AG": (-9.166, -21.535), "tg/AC": (-10.046, -22.591),
    "tt/AA": (-10.419, -27.683),
    "aA/TT": (-7.193, -19.723), "aC/TG": (-7.269, -18.336), "aG/TC": (-7.536, -18.387),
    "aT/TA": (-4.918, -12.943), "cA/GT": (-7.451, -18.380), "cC/GG": (-5.904, -11.904),
    "cG/GC": (-9.815, -23.491), "cT/GA": (-7.092, -16.825), "gA/CT": (-5.038, -11.656),
    "gC/CG": (-10.160, -24.651), "gG/CC": (-10.844, -26.580), "gT/CA": (-8.612, -22.327),
    "tA/AT": (-7.246, -19.738), "tC/AG": (-6.307, -15.515), "tG/AC": (-10.040, -25.744), 
    "tT/AA": (-6.372, -16.902), 
    "Aa/TT": (-6.908, -18.135), "Ac/TG": (-5.510, -11.824), "Ag/TC": (-9.000, -22.826),
    "At/TA": (-5.384, -13.537), "Ca/GT": (-7.142, -18.333), "Cc/GG": (-5.937, -12.335),
    "Cg/GC": (-10.876, -27.918), "Ct/GA": (-9.471, -25.070), "Ga/CT": (-7.756, -19.302),
    "Gc/CG": (-10.725, -25.511), "Gg/CC": (-8.943, -20.833), "Gt/CA": (-9.035, -22.742),
    "Ta/AT": (-5.609, -16.019), "Tc/AG": (-7.591, -19.031), "Tg/AC": (-6.335, -15.537),
    "Tt/AA": (-5.574, -14.149)
}

# Internal mismatch and inosine table (LNA)# Allawi & SantaLucia (1997), Biochemistry 36: 10581-10594
# Allawi & SantaLucia (1998), Biochemistry 37: 9435-9444
# Allawi & SantaLucia (1998), Biochemistry 37: 2170-2179
# Allawi & SantaLucia (1998), Nucl Acids Res 26: 2694-2701
# Peyret et al. (1999), Biochemistry 38: 3468-3477
# Watkins & SantaLucia (2005), Nucl Acids Res 33: 6258-6267
# Richard Owczarzy, Yong You, Christopher L. Groth, and Andrey V. Tataurov, Biochemistry, 50, 43, 9352-9367 2011.
# Confirmed correct on Mar 11, 2024
LNA_IMM1 = {
    "AG/TT": (1.0, 0.9), "AT/TG": (-2.5, -8.3), "CG/GT": (-4.1, -11.7),
    "CT/GG": (-2.8, -8.0), "GG/CT": (3.3, 10.4), "GG/TT": (5.8, 16.3),
    "GT/CG": (-4.4, -12.3), "GT/TG": (4.1, 9.5), "TG/AT": (-0.1, -1.7),
    "TG/GT": (-1.4, -6.2), "TT/AG": (-1.3, -5.3), "AA/TG": (-0.6, -2.3),
    "AG/TA": (-0.7, -2.3), "CA/GG": (-0.7, -2.3), "CG/GA": (-4.0, -13.2),
    "GA/CG": (-0.6, -1.0), "GG/CA": (0.5, 3.2), "TA/AG": (0.7, 0.7),
    "TG/AA": (3.0, 7.4),
    "AC/TT": (0.7, 0.2), "AT/TC": (-1.2, -6.2), "CC/GT": (-0.8, -4.5),
    "CT/GC": (-1.5, -6.1), "GC/CT": (2.3, 5.4), "GT/CC": (5.2, 13.5),
    "TC/AT": (1.2, 0.7), "TT/AC": (1.0, 0.7),
    "AA/TC": (2.3, 4.6), "AC/TA": (5.3, 14.6), "CA/GC": (1.9, 3.7),
    "CC/GA": (0.6, -0.6), "GA/CC": (5.2, 14.2), "GC/CA": (-0.7, -3.8),
    "TA/AC": (3.4, 8.0), "TC/AA": (7.6, 20.2),
    "AA/TA": (1.2, 1.7), "CA/GA": (-0.9, -4.2), "GA/CA": (-2.9, -9.8),
    "TA/AA": (4.7, 12.9), "AC/TC": (0.0, -4.4), "CC/GC": (-1.5, -7.2),
    "GC/CC": (3.6, 8.9), "TC/AC": (6.1, 16.4), "AG/TG": (-3.1, -9.5),
    "CG/GG": (-4.9, -15.3), "GG/CG": (-6.0, -15.8), "TG/AG": (1.6, 3.6),
    "AT/TT": (-2.7, -10.8), "CT/GT": (-5.0, -15.8), "GT/CT": (-2.2, -8.4),
    "TT/AT": (0.2, -1.5),
    "AI/TC": (-8.9, -25.5), "TI/AC": (-5.9, -17.4), "AC/TI": (-8.8, -25.4),
    "TC/AI": (-4.9, -13.9), "CI/GC": (-5.4, -13.7), "GI/CC": (-6.8, -19.1),
    "CC/GI": (-8.3, -23.8), "GC/CI": (-5.0, -12.6),
    "AI/TA": (-8.3, -25.0), "TI/AA": (-3.4, -11.2), "AA/TI": (-0.7, -2.6),
    "TA/AI": (-1.3, -4.6), "CI/GA": (2.6, 8.9), "GI/CA": (-7.8, -21.1),
    "CA/GI": (-7.0, -20.0), "GA/CI": (-7.6, -20.2),
    "AI/TT": (0.49, -0.7), "TI/AT": (-6.5, -22.0), "AT/TI": (-5.6, -18.7),
    "TT/AI": (-0.8, -4.3), "CI/GT": (-1.0, -2.4), "GI/CT": (-3.5, -10.6),
    "CT/GI": (0.1, -1.0), "GT/CI": (-4.3, -12.1),
    "AI/TG": (-4.9, -15.8), "TI/AG": (-1.9, -8.5), "AG/TI": (0.1, -1.8),
    "TG/AI": (1.0, 1.0), "CI/GG": (7.1, 21.3), "GI/CG": (-1.1, -3.2),
    "CG/GI": (5.8, 16.9), "GG/CI": (-7.6, -22.0),
    "AI/TI": (-3.3, -11.9), "TI/AI": (0.1, -2.3), "CI/GI": (1.3, 3.0),
    "GI/CI": (-0.5, -1.3),
    #aA mismatch
    "aa/AT": (-3.826, -13.109), "ac/AG": (-2.367, -7.322), "ag/AC": (-4.849, -13.007),
    "at/AA": (-5.049, -17.514), "aa/TA": (-4.229, -15.160), "ca/GA": (-5.878, -17.663),
    "ga/CA": (-8.558, -23.976), "ta/AA": (2.074, 3.446),
    #cC mismatch
    "ca/CT": (2.218, 4.750), "cc/CG": (1.127, 1.826), "cg/CC": (-10.903, -32.025),
    "ct/CA": (-2.053, -10.517), "ac/TC": (1.065, -1.403), "cc/GC": (-9.522, -27.024),
    "gc/CC": (-4.767, -14.897), "tc/AC": (4.114, 9.258),
    #gG mismatch
    "ga/GT": (-2.920, -9.387), "gc/GG": (-8.139, -21.784), "gg/GC": (-5.149, -12.508),
    "gt/GA": (-8.991, -27.311), "ag/TG": (-4.980, -15.426), "cg/GG": (-4.441, -12.158),
    "gg/CG": (-13.505, -36.021), "tg/AG": (-2.775, -9.286),
    #tT mismatch
    "ta/TT": (-3.744, -12.149), "tc/TG": (-4.387, -13.520), "tg/TC": (-6.346, -16.629),
    "tt/TA": (-7.697, -25.049), "at/TT": (-4.207, -14.307), "ct/GT": (-8.176, -22.962),
    "gt/CT": (-7.241, -20.622), "tt/AT": (-2.051, -7.055), 
    #aC mismatch
    "aa/CT": (-1.362, -5.551), "ac/CG": (-1.759, -6.511), "ag/CC": (-6.549, -18.073),
    "at/CA": (-3.563, -14.105), "aa/TC": (-2.078, -10.088), "ca/GC": (-5.868, -16.952),
    "ga/CC": (-8.477, -24.565), "ta/AC": (2.690, 4.965),
    #cA mismatch
    "ca/AT": (-9.844, -29.673), "cc/AG": (-3.761, -11.204), "cg/AC": (-9.845, -27.316),
    "ct/AA": (-3.389, -12.517), "ac/TA": (0.753, -0.503), "cc/GA": (-12.714, -35.555),
    "gc/CA": (-12.658, -35.729), "tc/AA": (-1.719, -7.023),
    #aG mismatch
    "aa/GT": (2.193, 4.374), "ac/GG": (-8.453, -22.672), "ag/GC": (-1.164, -2.532),
    "at/GA": (-7.418, -24.066), "aa/TG": (-1.963, -9.013), "ca/GG": (-8.712, -23.779),
    "ga/CG": (-7.875, -21.661), "ta/AG": (3.207, 7.156),
    #gA mismatch
    "ga/AT": (-2.914, -9.402), "gc/AG": (-9.131, -25.347), "gg/AC": (-2.154, -3.871),
    "gt/AA": (-8.515, -26.313), "ag/TA": (-6.691, -21.148), "cg/GA": (-3.960, -10.588),
    "gg/CA": (-12.898, -34.656), "tg/AA": (0.334, -0.440),
    #cT mismatch
    "ca/TT": (0.382, -0.579), "cc/TG": (-2.716, -8.000), "cg/TC": (-10.363, -29.315),
    "ct/TA": (-5.783, -20.173), "ac/TT": (-0.692, -5.278), "cc/GT": (-10.299, -28.503),
    "gc/CT": (-9.062, -26.356), "tc/AT": (2.073, 3.968),
    #tC mismatch
    "ta/CT": (-5.485, -17.347), "tc/CG": (1.451, 1.556), "tg/CC": (-7.213, -20.128),
    "tt/CA": (-2.397, -11.371), "at/TC": (-0.633, -5.801), "ct/GC": (-6.868, -21.000),
    "gt/CC": (-5.853, -16.643), "tt/AC": (0.211, -1.446),
    #GT mismatch
    "ga/TT": (-5.551, -15.398), "gc/TG": (-14.943, -40.148), "gg/TC": (-8.110, -18.349),
    "gt/TA": (-14.213, -40.041), "ag/TT": (-7.130, -20.786), "cg/GT": (-14.862, -39.430),
    "gg/CT": (-14.622, -37.510), "tg/AT": (-6.703, -18.111),
    #TG mismatch
    "ta/GT": (-4.612, -14.039), "tc/GG": (-9.798, -26.406), "tg/GC": (-4.519, -11.065),
    "tt/GA": (-4.523, -15.693), "at/TG": (-2.364, -8.834), "ct/GG": (-11.396, -30.732), 
    "gt/CG": (-6.233, -15.933), "tt/AG": (-2.960, -9.305)}


def is_mismatch(bases):
    pair = [("A", "T"), ("G", "C"), ("T", "A"), ("C", "G")]
    if bases in pair:
        return False
    else:
        return True
    
def Tm_NN(
    seq,
    check=True,
    strict=True,
    c_seq=None,
    shift=0,
    nn_table=None,
    tmm_table=None,
    imm_table=None,
    de_table=None,
    dnac1=25,
    dnac2=25,
    selfcomp=False,
    Na=50,
    K=0,
    Tris=0,
    Mg=0,
    dNTPs=0,
    saltcorr=5,
):
    """Return the Tm using nearest neighbor thermodynamics.

    Arguments:
     - seq: The primer/probe sequence as string or Biopython sequence object.
       For RNA/DNA hybridizations seq must be the RNA sequence.
     - c_seq: Complementary sequence. The sequence of the template/target in
       3'->5' direction. c_seq is necessary for mismatch correction and
       dangling-ends correction. Both corrections will automatically be
       applied if mismatches or dangling ends are present. Default=None.
     - shift: Shift of the primer/probe sequence on the template/target
       sequence, e.g.::

                           shift=0       shift=1        shift= -1
        Primer (seq):      5' ATGC...    5'  ATGC...    5' ATGC...
        Template (c_seq):  3' TACG...    3' CTACG...    3'  ACG...

       The shift parameter is necessary to align seq and c_seq if they have
       different lengths or if they should have dangling ends. Default=0
     - table: Thermodynamic NN values, eight tables are implemented:
       For DNA/DNA hybridizations:

        - DNA_NN1: values from Breslauer et al. (1986)
        - DNA_NN2: values from Sugimoto et al. (1996)
        - DNA_NN3: values from Allawi & SantaLucia (1997) (default)
        - DNA_NN4: values from SantaLucia & Hicks (2004)

       For RNA/RNA hybridizations:

        - RNA_NN1: values from Freier et al. (1986)
        - RNA_NN2: values from Xia et al. (1998)
        - RNA_NN3: valuse from Chen et al. (2012)

       For RNA/DNA hybridizations:

        - R_DNA_NN1: values from Sugimoto et al. (1995)
          Note that ``seq`` must be the RNA sequence.

       Use the module's maketable method to make a new table or to update one
       one of the implemented tables.
     - tmm_table: Thermodynamic values for terminal mismatches.
       Default: DNA_TMM1 (SantaLucia & Peyret, 2001)
     - imm_table: Thermodynamic values for internal mismatches, may include
       insosine mismatches. Default: DNA_IMM1 (Allawi & SantaLucia, 1997-1998;
       Peyret et al., 1999; Watkins & SantaLucia, 2005)
     - de_table: Thermodynamic values for dangling ends:

        - DNA_DE1: for DNA. Values from Bommarito et al. (2000) (default)
        - RNA_DE1: for RNA. Values from Turner & Mathews (2010)

     - dnac1: Concentration of the higher concentrated strand [nM]. Typically
       this will be the primer (for PCR) or the probe. Default=25.
     - dnac2: Concentration of the lower concentrated strand [nM]. In PCR this
       is the template strand which concentration is typically very low and may
       be ignored (dnac2=0). In oligo/oligo hybridization experiments, dnac1
       equals dnac1. Default=25.
       MELTING and Primer3Plus use k = [Oligo(Total)]/4 by default. To mimic
       this behaviour, you have to divide [Oligo(Total)] by 2 and assign this
       concentration to dnac1 and dnac2. E.g., Total oligo concentration of
       50 nM in Primer3Plus means dnac1=25, dnac2=25.
     - selfcomp: Is the sequence self-complementary? Default=False. If 'True'
       the primer is thought binding to itself, thus dnac2 is not considered.
     - Na, K, Tris, Mg, dNTPs: See method 'Tm_GC' for details. Defaults: Na=50,
       K=0, Tris=0, Mg=0, dNTPs=0.
     - saltcorr: See method 'Tm_GC'. Default=5. 0 means no salt correction.

    """
    # Set defaults
    # if not nn_table:
    #     nn_table = DNA_NN3
    # if not tmm_table:
    #     tmm_table = DNA_TMM1
    # if not imm_table:
    #     imm_table = DNA_IMM1
    # if not de_table:
    #     de_table = DNA_DE1

    seq = str(seq)
    if not c_seq:
        # c_seq must be provided by user if dangling ends or mismatches should
        # be taken into account. Otherwise take perfect complement.
        c_seq = Seq.Seq(seq).complement()
    c_seq = str(c_seq)
    
    tmp_seq = seq
    tmp_cseq = c_seq
    delta_h = 0
    delta_s = 0
    d_h = 0  # Names for indexes
    d_s = 1  # 0 and 1

    # # Dangling ends?
    # if shift or len(seq) != len(c_seq):
    #     # Align both sequences using the shift parameter
    #     if shift > 0:
    #         tmp_seq = "." * shift + seq
    #     if shift < 0:
    #         tmp_cseq = "." * abs(shift) + c_seq
    #     if len(tmp_cseq) > len(tmp_seq):
    #         tmp_seq += (len(tmp_cseq) - len(tmp_seq)) * "."
    #     if len(tmp_cseq) < len(tmp_seq):
    #         tmp_cseq += (len(tmp_seq) - len(tmp_cseq)) * "."
    #     # Remove 'over-dangling' ends
    #     while tmp_seq.startswith("..") or tmp_cseq.startswith(".."):
    #         tmp_seq = tmp_seq[1:]
    #         tmp_cseq = tmp_cseq[1:]
    #     while tmp_seq.endswith("..") or tmp_cseq.endswith(".."):
    #         tmp_seq = tmp_seq[:-1]
    #         tmp_cseq = tmp_cseq[:-1]
    #     # Now for the dangling ends
    #     if tmp_seq.startswith(".") or tmp_cseq.startswith("."):
    #         left_de = tmp_seq[:2] + "/" + tmp_cseq[:2]
    #         try:
    #             delta_h += de_table[left_de][d_h]
    #             delta_s += de_table[left_de][d_s]
    #         except KeyError:
    #             _key_error(left_de, strict)
    #         tmp_seq = tmp_seq[1:]
    #         tmp_cseq = tmp_cseq[1:]
    #     if tmp_seq.endswith(".") or tmp_cseq.endswith("."):
    #         right_de = tmp_cseq[-2:][::-1] + "/" + tmp_seq[-2:][::-1]
    #         try:
    #             delta_h += de_table[right_de][d_h]
    #             delta_s += de_table[right_de][d_s]
    #         except KeyError:
    #             _key_error(right_de, strict)
    #         tmp_seq = tmp_seq[:-1]
    #         tmp_cseq = tmp_cseq[:-1]

    # Now for terminal mismatches
    # left_tmm = tmp_cseq[:2][::-1] + "/" + tmp_seq[:2][::-1]
    # if left_tmm in tmm_table:
    #     delta_h += tmm_table[left_tmm][d_h]
    #     delta_s += tmm_table[left_tmm][d_s]
    #     tmp_seq = tmp_seq[1:]
    #     tmp_cseq = tmp_cseq[1:]
    # right_tmm = tmp_seq[-2:] + "/" + tmp_cseq[-2:]
    # if right_tmm in tmm_table:
    #     delta_h += tmm_table[right_tmm][d_h]
    #     delta_s += tmm_table[right_tmm][d_s]
    #     tmp_seq = tmp_seq[:-1]
    #     tmp_cseq = tmp_cseq[:-1]

    # Now everything 'unusual' at the ends is handled and removed and we can
    # look at the initiation.
    # One or several of the following initiation types may apply:

    # Type: General initiation value
    delta_h += nn_table["init"][d_h]
    delta_s += nn_table["init"][d_s]

    # Type: Duplex with no (allA/T) or at least one (oneG/C) GC pair
    if SeqUtils.gc_fraction(seq) == 0:
        delta_h += nn_table["init_allA/T"][d_h]
        delta_s += nn_table["init_allA/T"][d_s]
    else:
        delta_h += nn_table["init_oneG/C"][d_h]
        delta_s += nn_table["init_oneG/C"][d_s]

    # Type: Penalty if 5' end is T
    # actually 0
    if seq.startswith("T") or seq.startswith("t"):
        delta_h += nn_table["init_5T/A"][d_h]
        delta_s += nn_table["init_5T/A"][d_s]
    if seq.endswith("A") or seq.endswith("a"):
        delta_h += nn_table["init_5T/A"][d_h]
        delta_s += nn_table["init_5T/A"][d_s]

    # Type: Different values for G/C or A/T terminal basepairs
    # assuming that the initiation parameter of LNA is the same as that of DNA
    for i in [0, -1]:
        if not is_mismatch((seq[i].upper(), c_seq[i].upper())):
            ends = seq[i]
            AT = ends.count("A") + ends.count("T") + ends.count("a") + ends.count("t")
            GC = ends.count("G") + ends.count("C") + ends.count("g") + ends.count("c")
            delta_h += nn_table["init_A/T"][d_h] * AT
            delta_s += nn_table["init_A/T"][d_s] * AT
            delta_h += nn_table["init_G/C"][d_h] * GC
            delta_s += nn_table["init_G/C"][d_s] * GC

    # Finally, the 'zipping'
    for basenumber in range(len(tmp_seq) - 1):
        neighbors = (
            tmp_seq[basenumber : basenumber + 2]
            + "/"
            + tmp_cseq[basenumber : basenumber + 2]
        )
        if neighbors in imm_table:
            delta_h += imm_table[neighbors][d_h]
            delta_s += imm_table[neighbors][d_s]
        elif neighbors[::-1] in imm_table:
            delta_h += imm_table[neighbors[::-1]][d_h]
            delta_s += imm_table[neighbors[::-1]][d_s]
        elif neighbors in nn_table:
            delta_h += nn_table[neighbors][d_h]
            delta_s += nn_table[neighbors][d_s]
        elif neighbors[::-1] in nn_table:
            delta_h += nn_table[neighbors[::-1]][d_h]
            delta_s += nn_table[neighbors[::-1]][d_s]
        else:
            # We haven't found the key...
            _key_error(neighbors, strict)

    k = (dnac1 - (dnac2 / 2.0)) * 1e-9
    if selfcomp:
        k = dnac1 * 1e-9
        delta_h += nn_table["sym"][d_h]
        delta_s += nn_table["sym"][d_s]
    R = 1.987  # universal gas constant in Cal/degrees C*Mol
    if saltcorr:
        corr = SeqUtils.MeltingTemp.salt_correction(
            Na=Na, K=K, Tris=Tris, Mg=Mg, dNTPs=dNTPs, method=saltcorr, seq=seq
        )
    if saltcorr == 5:
        delta_s += corr
    melting_temp = (1000 * delta_h) / (delta_s + (R * (math.log(k)))) - 273.15
    if saltcorr in (1, 2, 3, 4):
        melting_temp += corr
    if saltcorr in (6, 7):
        # Tm = 1/(1/Tm + corr)
        melting_temp = 1 / (1 / (melting_temp + 273.15) + corr) - 273.15

    return melting_temp, delta_h, delta_s



def complementaly(base):
    dic = {'A':'T', 'T':'A', 'G':'C', 'C':'G'}
    return dic[base]

def base2num(base):
    dic = {'T':0, 'A':1, 'C':2, 'G':3}
    return dic[base]

def blocker_seq(seq_char_list, pos, blocker_length, max_offset, err_offset, change_to):
    """Return the blocker sequence in the form of the list of charactors. Direction: 5' -> 3'

    Arguments:
     - seq_char_list: The original template sequence in the form of list of charactors. Direction: 3'-> 5'
     - pos: The error position that is counted from the left side of teh original template sequence. start from zero. 
     - blocker_length: Total length of the blocker
     - max_offset: Maximum length of DNA region on the left side of the LNA region of the blocker.  
                   example of this offset:
                   e.g, AATTcgcg... : offset = 4, AATTCgcgt... : offset = 5.
                   max_offset defines the maximum length of offset. 
     - err_offset: error position counting from the left side of the LNA region. start from zero.  
                   e.g, 5'AAGCcgcaTATA3' if g is faced with the mutation, err_offset = 1. 
                   e.g, 5'AAGCcgcaTATA3' if a is faced with the mutation, err_offset = 3. 
     - change_to: the template base at pos is changed to change_to.

     e.g., err_offset = 1
     0123456789.....
     blocker length = 14
     ATGCAggacTCTCT
     <- -> |
     offset         length of offset canbe smaller than max_offset, but not more than max_offset.
    """
    num_of_LNA = 4
    seq_char_list[pos] = change_to
    blocker_seq_char_list = []
    LNA_starts_at = pos - err_offset
    if LNA_starts_at <= 0: #LNA region is the most 5' side of the blocker
        offset = 0
        for i in range(num_of_LNA):
            blocker_seq_char_list.append(complementaly(seq_char_list[i]).lower())
        for i in range(blocker_length - num_of_LNA):
            blocker_seq_char_list.append(complementaly(seq_char_list[num_of_LNA+i]))
    elif LNA_starts_at < max_offset: # there is offset region smaller than max_offset. 
        offset = LNA_starts_at
        for i in range(offset):
            blocker_seq_char_list.append(complementaly(seq_char_list[i]))
        for i in range(num_of_LNA):
            blocker_seq_char_list.append(complementaly(seq_char_list[offset+i]).lower())
        for i in range(blocker_length - offset - num_of_LNA):
            blocker_seq_char_list.append(complementaly(seq_char_list[offset+num_of_LNA+i]))
    elif LNA_starts_at >= max_offset: # offset length is equal to max_offset.
        offset = max_offset
        for i in range(offset):
            blocker_seq_char_list.append(complementaly(seq_char_list[pos-err_offset-offset+i]))
        for i in range(num_of_LNA):
            blocker_seq_char_list.append(complementaly(seq_char_list[pos-err_offset+i]).lower())
        for i in range(blocker_length - offset - num_of_LNA):
            blocker_seq_char_list.append(complementaly(seq_char_list[pos-err_offset+num_of_LNA+i]))
    return blocker_seq_char_list

def create_error(seq_char_list, pos, blocker_length, max_offset, err_offset, change_to):
    """Return the mutated template sequence in the form of the list of charactors. Direction: 3' -> 5'

    Arguments:
     - seq_char_list: The original template sequence in the form of list of charactors. Direction: 3'-> 5'
     - pos: The error position that is counted from the left side of teh original template sequence. start from zero. 
     - blocker_length: Total length of the blocker
     - max_offset: Maximum length of DNA region on the left side of the LNA region of the blocker.
                   example of this offset:
                   e.g, AATTcgcg... : offset = 4, AATTCgcgt... : offset = 5.
                   max_offset defines the maximum length of offset. 
     - err_offset: error position counting from the left side of the LNA region. start from zero.  
                   e.g, 5'AAGCcgcaTATA3' if g is faced with the mutation, err_offset = 1. 
                   e.g, 5'AAGCcgcaTATA3' if a is faced with the mutation, err_offset = 3. 
     - change_to: the template base at pos is changed to change_to.
    """
    seq_char_list[pos] = change_to
    may_starts_at = pos - err_offset - max_offset
    if may_starts_at <= 0: # for sequence in the most left side region.
        return seq_char_list[0:blocker_length]
    else: # for other sequence 
        return seq_char_list[may_starts_at:may_starts_at+blocker_length]
    
def error_name(n, base):
    return 'W('+str(n) +',' + base + ')'

def blocker_name(n, base):
    return 'B('+str(n) +',' + base + ')'


def calcKij(params, template, primer_length):
    
    # template: sequence of the template in the order of 5' -> 3'. The primer binds to the 3' end.
    template = template[::-1] # convert the sequence to 3' -> 5'
    
    original_seq_char_list = [*template]
    
    blocker_length = params['blocker length']
    
    err_offset = 1
    max_offset = int(np.round(blocker_length/2, 0)) - (err_offset + 1)
    bases = ['T', 'A', 'C', 'G']
    
    Tab = params['temperature'] + 273.15    
    
    kBT = 1.38E-23 * Tab
    NA = 6.02E23
    RT = kBT * NA
    
    dG_array = np.zeros((primer_length, 4, 4))

    for i in range(primer_length):
        seq_list, cseq_list = [], []
        
        for j in range(4):
            error_seq_char_list = create_error(copy.deepcopy(original_seq_char_list), i, blocker_length, max_offset, err_offset, bases[j])
            blocker_seq_char_list = blocker_seq(copy.deepcopy(original_seq_char_list), i, blocker_length, max_offset, err_offset, bases[j]) 
            
            cseq_list.append(''.join(error_seq_char_list))
            seq_list.append(''.join(blocker_seq_char_list))
            
        dG = np.zeros((4, 4))
        for j in range(4):
            for k in range(4):
                _, dH, dS = Tm_NN(seq = seq_list[k], c_seq = cseq_list[j], check=False, nn_table=LNA_NN, imm_table = LNA_IMM1, Na=50, Mg=1.5, dNTPs=0.2,dnac1=10000, dnac2=2.5)
                dG[j, k] = dH - Tab*dS/1000  # j: W, k: B, 
                
        dG_array[i] = dG

    dG_array *= 4184 # kcal/mol to J/mol
    dG_array /= RT
    Kij = np.exp(dG_array)
    
    return Kij


def error_mean(B, a, p_k, eta_k_phi):    
    return np.sum(p_k * eta_k_phi / (np.dot(a, B) + 1.0))


def error_max(B, a, eta_k_phi): 
    return np.max([((eta_k_phi[k])/(np.dot(a[k], B) + 1.0)) for k in range(len(B))])
    

# minimization of error fraction and find the optimal blocker concentrationss
def replicator(a, eta_k_phi, p_k, params, ifMean):
    dt = 0.005
    
    Btot = params['Btot'] # uM
    maxCount = params['max iteration count']
    error_tolerance = params['error tolerance']
    
    N = len(p_k)
    B = np.array([Btot/N]*N)

    count = 0
    error_previous = 1
    while (count < maxCount):     
        if ifMean:
            Bcom = np.dot(a, B)

            for i in range(N): 
                sumk = np.sum([p_k[k] * eta_k_phi[k] / (Bcom[k] + 1.0) / (Bcom[k] + 1.0) * (a[k][i] - Bcom[k] / Btot) for k in range(N)])
                B[i] += B[i] * dt * sumk

        else:
            each_error = [((eta_k_phi[k])/(np.dot(a[k], B) + 1.0)) for k in range(N)]
                    
            Kaste = np.argmax(each_error)
            Bcom = np.dot(a[Kaste], B)

            for i in range(N):
                tmp = ((eta_k_phi[Kaste])/(Bcom+1.0)**2)*(a[Kaste][i]-Bcom/Btot)
                B[i] = B[i]*(1.0 + dt*tmp)
            
        if count % 5000 == 0:
            error = error_mean(B ,a ,p_k ,eta_k_phi) if ifMean else max(each_error)
            if params['report convergence']:
                print("Count, error fraction = ", count , error)   

            if np.abs((error-error_previous)/error_previous) < error_tolerance:
                break

            error_previous = error
        count+=1
        
    B_optimal = B
    B_zero = [0]*N
    B_uniform = [Btot/N]*N

    error_zero = error_mean(B_zero, a, p_k, eta_k_phi) if ifMean else error_max(B_zero, a, eta_k_phi)
    error_uniform = error_mean(B_uniform, a, p_k, eta_k_phi) if ifMean else error_max(B_uniform, a, eta_k_phi)
    error_optimal = error_mean(B_optimal, a, p_k, eta_k_phi) if ifMean else error_max(B_optimal, a, eta_k_phi)

    print("Parameters:")   
    print("  Btot = ", Btot, "uM")
    print("  p_k = ", np.round(p_k,3))
    print("  K_ij = ", np.round(1/a, 3), 'uM')
    print("  Error fractions of W_i without blockers = ", np.round(eta_k_phi,3))
    print("")
    print("Results:")
    
    print("  Error fraction w/o blockers = {:.3g}".format(error_zero))    
    print("  Error fraction with uniform blockers =  {:.3g}".format(error_uniform))
    print("  Error fraction with optimal blockers =  {:.3g}".format(error_optimal))
    print("  Ratio (None/Optimal) = {:.3g}".format(error_zero/error_optimal))
    print("  Optimal blocker conc = ", np.round(B_optimal,3), "uM")
    print("  Iteration count = ",count)
    print("------------")
    print("")
    print("")
     

    return error_optimal, B_optimal, eta_k_phi


def list_mutations(R, W, p): # R and W are given in the order of 5' -> 3'
    mutPos, mutFrom, mutTo = [], [], []
    
    length = len(R)
    
    p2, W2 = [], []
    for w, prob in zip(W, p):    
        count = 0
        for w_index in range(length):
            if R[w_index] != w[w_index]:
                count+=1
                pos = w_index
        
        if count==1:
            mutPos.append(length - pos - 1)   # position is measured from the 3' end. pos = 0 at the 3'-end.
            mutFrom.append(R[pos])
            mutTo.append(w[pos])
            p2.append(prob)
            W2.append(w)
            
    return mutPos, mutFrom, mutTo, p2, W2


def calc_epsilon_tilde(pos, length, params):
    # pos is counted from the 3' end
    
    p, q, r, s, u = params['p'], params['q'], params['r'], params['s'], params['u']
    
    return p*(1 + np.tanh(q*(pos - r)))*(1 - np.tanh(s*(pos - u - length + 20)))


def optimization(params, template, W, p, ifMean):
    """
Returns error_optimal, B_optimal, eta_k_phi, aij.

Arguments:
params  – parameters, must be dictionary
template   – template sequence, must be string
W   – comtaminating sequences, must be array of string
p   – probability of contamination of W sequences, must be array of float
ifMean - True if mean error is to be minimized. False if maximum error is to be minimized.
"""

    print("------------")
    
    # normalization
    template = template.replace(' ','').upper()  
    
    W = [w.replace(' ','').upper() for w in W] # eliminate spaces.


    length = len(W[0])
    for w in W:
        if len(w) != length:
            raise ValueError("All W sequences should have the same length.")
        if len(w) > len(template):
            raise ValueError("Length of W should not be longer than the length of template.")
            
    length = len(W[0])
    
    blocker_length = params['blocker length']
    
    if len(template) <= length + int(blocker_length/2):
         raise ValueError("Template length (now {:d}) should be equal or greater than primer length ({:d}) + blocker length/2 ({:d})".format(len(template), length, int(blocker_length/2)))
    
    

    R = template[-length:] # pick "length" bp from the 3' end.


    mut_pos, mut_from, mut_to, p_i, W = list_mutations(R, W, p)
    N = len(mut_pos)
            
    print('R: 5\'-', R, '-3\'')
    print('W: 5\'-', W, '-3\'')
    mut_str = [str(mut_pos[k]) + mut_from[k] +" -> "+mut_to[k] for k in range(N)]
    print('mutations: ', mut_str, ' (Position starts from 3\' end)')
    print('')


    Kij = 1E6 * calcKij(params, template, length)   # Kij[pos, W, B]. unit: uM. pos =0 at the 3' end.
    

    # Calibration of KD using fitting curve
    
    # K' = np.exp(p*np.log(K)+q)
    # here, K and K' are in the unit of M.

    a, b =  params['calib_a'], params['calib_b'] # 0.6185290898147885, -4.3830120235322925

    Kij = np.exp(a*np.log(Kij*1E-6) + b) * 1E6  # unit is uM

    aij = 1.0/ Kij
    aij[aij <= 1/200] = 0
    
    epsilon0 = calc_epsilon_tilde(np.array(mut_pos), length, params) # error fraction without blocker
    
    aij_pick = np.zeros((N, N))
    for w_index in range(N): # W
        for b_index in range(N): # B
            aij_pick[w_index, b_index] = aij[mut_pos[w_index], base2num(mut_to[w_index]), base2num(mut_to[b_index])] if  mut_pos[w_index] == mut_pos[b_index] else 0

    error_optimal, B_optimal, eta_k_phi = replicator(aij_pick, epsilon0, p_i, params, ifMean)

    return error_optimal, B_optimal, eta_k_phi, aij_pick