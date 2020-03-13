#!/usr/bin/env python
# -*- coding: utf-8 -*-

valid_levels_and_solutions = {
# level_code: tuple(solution_codes)

# 1-1 Of Pancakes and Spaceships
'''H4sIAD7sal4A/5WPy2rEMAxFfyVonYCT1ZB8xBS6a+lCdeTE1CMby4amIf322jOl9LHqRnAPl8
vRDpZDTt2bZxIYd1D1XFmJjztcvCOdHcEI59dtIZ7O72qY+v40qGnoT0pBC9pnTjD2w/F0HC34
nP5u/mtI1ZkVpXMYF+pugzAadEItPHueKXafZXVrCrH4+NWpyGShn0SCsyn9gokcBR+/47SFKh
pJCKNeixnj5apumjtkjS8kDfLc3AfUJKsNUjqY01oV4MEuJc7WGFteTluRPD4A+MBbU2oBAAA=''':
(
'''SOLUTION:Of Pancakes and Spaceships,Zig,107-1-7
COMPONENT:'custom-research-reactor',2,0,''
MEMBER:'instr-start',-90,0,128,1,2,0,0
MEMBER:'instr-start',180,0,32,0,7,0,0
MEMBER:'instr-arrow',0,0,64,1,1,0,0
MEMBER:'instr-arrow',180,0,64,6,1,0,0
MEMBER:'instr-grab',-1,2,128,6,1,0,0
MEMBER:'instr-input',-1,0,128,1,1,0,0
MEMBER:'instr-grab',-1,1,128,2,1,0,0
MEMBER:'instr-output',-1,0,128,5,1,0,0
MEMBER:'instr-rotate',-1,0,128,4,1,0,0
PIPE:0,4,1
PIPE:1,4,2''',
'''SOLUTION:Of Pancakes and Spaceships,Zig,45-1-14,Cycles
COMPONENT:'empty-research-reactor',2,0,''
MEMBER:'instr-start',-90,0,128,2,5,0,0
MEMBER:'instr-start',-90,0,32,2,2,0,0
MEMBER:'instr-arrow',0,0,64,2,1,0,0
MEMBER:'instr-arrow',180,0,64,6,1,0,0
MEMBER:'instr-grab',-1,1,128,2,1,0,0
MEMBER:'instr-grab',-1,2,128,6,1,0,0
MEMBER:'instr-arrow',0,0,16,2,1,0,0
MEMBER:'instr-arrow',180,0,16,6,1,0,0
MEMBER:'instr-grab',-1,1,32,2,1,0,0
MEMBER:'instr-grab',-1,2,32,6,1,0,0
MEMBER:'instr-output',-1,0,32,3,1,0,0
MEMBER:'instr-input',-1,0,32,4,1,0,0
MEMBER:'instr-input',-1,0,128,2,4,0,0
MEMBER:'instr-rotate',-1,0,32,5,1,0,0
MEMBER:'instr-rotate',-1,0,128,4,1,0,0
MEMBER:'instr-output',-1,0,128,5,1,0,0
PIPE:0,4,1
PIPE:1,4,2''',
),

# 1-4 An Introduction to Bonding
'''H4sIAMa7aV4A/3WPQWvDMAyF/0rQaYME7FIYc07rIdDzbh07eImSGFwr2HKhC/nvs5sxtoZdBP
p47+lpBuOmyNUnOQygZhB53Fha32Y4k8U2WgQFr8Ze0NcvQy3l/kkIKKGl6BiU3C3vSwnyf29j
I3njsG6S+XnjTWaKvC2yuV48HB+LNazDVKW5dZGi3t2lipw56lBZ7Qes1nRQvbYBS/gg16Gvvs
X7VRnQBfI/moz6GPAvCZM1zHeQ0eJE/jfm65RbewyofTumZk6fMzk69tTFlg25gqk4pCrGDUmg
I4/5PpxMXjvT9yY9z1dQYvkCUjns8KkBAAA=''':
(
'''SOLUTION:An Introduction to Bonding,Zig,154-1-10,Symbols
COMPONENT:'tutorial-research-reactor-2',2,0,''
MEMBER:'instr-start',180,0,128,3,5,0,0
MEMBER:'instr-start',180,0,32,1,6,0,0
MEMBER:'feature-bonder',-1,0,1,1,1,0,0
MEMBER:'feature-bonder',-1,0,1,1,2,0,0
MEMBER:'feature-bonder',-1,0,1,1,4,0,0
MEMBER:'feature-bonder',-1,0,1,1,5,0,0
MEMBER:'instr-grab',-1,1,128,1,5,0,0
MEMBER:'instr-arrow',-90,0,64,1,5,0,0
MEMBER:'instr-arrow',0,0,64,1,2,0,0
MEMBER:'instr-arrow',90,0,64,6,2,0,0
MEMBER:'instr-arrow',180,0,64,6,5,0,0
MEMBER:'instr-input',-1,0,128,1,3,0,0
MEMBER:'instr-input',-1,1,128,2,5,0,0
MEMBER:'instr-bond',-1,0,128,1,2,0,0
MEMBER:'instr-output',-1,0,32,0,6,0,0
MEMBER:'instr-grab',-1,2,128,6,2,0,0
PIPE:0,4,1
PIPE:1,4,2''',
'''SOLUTION:An Introduction to Bonding,Zig,74-1-40,Cycles
COMPONENT:'tutorial-research-reactor-2',2,0,''
MEMBER:'instr-start',-90,0,128,1,7,0,0
MEMBER:'instr-start',0,0,32,0,0,0,0
MEMBER:'feature-bonder',-1,0,1,1,1,0,0
MEMBER:'feature-bonder',-1,0,1,1,2,0,0
MEMBER:'feature-bonder',-1,0,1,1,4,0,0
MEMBER:'feature-bonder',-1,0,1,1,5,0,0
MEMBER:'instr-grab',-1,1,32,1,1,0,0
MEMBER:'instr-arrow',0,0,16,1,5,0,0
MEMBER:'instr-arrow',90,0,64,6,3,0,0
MEMBER:'instr-grab',-1,2,128,6,3,0,0
MEMBER:'instr-grab',-1,1,32,1,4,0,0
MEMBER:'instr-arrow',90,0,16,1,4,0,0
MEMBER:'instr-arrow',-90,0,16,3,5,0,0
MEMBER:'instr-arrow',180,0,16,3,4,0,0
MEMBER:'instr-arrow',-90,0,64,4,4,0,0
MEMBER:'instr-grab',-1,1,128,4,4,0,0
MEMBER:'instr-rotate',-1,0,128,5,3,0,0
MEMBER:'instr-grab',-1,2,32,3,4,0,0
MEMBER:'instr-rotate',-1,1,32,3,5,0,0
MEMBER:'instr-arrow',90,0,16,2,3,0,0
MEMBER:'instr-arrow',180,0,16,2,4,0,0
MEMBER:'instr-bond',-1,1,32,1,5,0,0
MEMBER:'instr-arrow',0,0,64,4,3,0,0
MEMBER:'instr-arrow',180,0,64,6,4,0,0
MEMBER:'instr-arrow',90,0,64,6,2,0,0
MEMBER:'instr-input',-1,1,32,2,3,0,0
MEMBER:'instr-input',-1,0,32,1,0,0,0
MEMBER:'instr-grab',-1,1,128,1,5,0,0
MEMBER:'instr-bond',-1,0,128,1,2,0,0
MEMBER:'instr-input',-1,1,128,5,4,0,0
MEMBER:'instr-arrow',0,0,64,1,2,0,0
MEMBER:'instr-bond',-1,0,128,5,2,0,0
MEMBER:'instr-input',-1,1,128,1,6,0,0
MEMBER:'instr-arrow',90,0,16,1,0,0,0
MEMBER:'instr-bond',-1,0,32,1,2,0,0
MEMBER:'instr-bond',-1,0,32,1,3,0,0
MEMBER:'instr-input',-1,0,128,1,4,0,0
MEMBER:'instr-input',-1,0,128,1,3,0,0
MEMBER:'instr-arrow',180,0,16,1,2,0,0
MEMBER:'instr-arrow',90,0,16,0,2,0,0
MEMBER:'instr-arrow',0,0,16,0,3,0,0
MEMBER:'instr-input',-1,0,128,2,2,0,0
MEMBER:'instr-bond',-1,0,32,2,4,0,0
MEMBER:'instr-input',-1,0,128,4,3,0,0
MEMBER:'instr-input',-1,0,128,4,2,0,0
MEMBER:'instr-output',-1,0,32,2,5,0,0
PIPE:0,4,1
PIPE:1,4,2''',
),

# 2-1 Double Bonds
'''H4sIAHDCal4A/22PywrCMBBFf0Vm3UJaXEi6E/fuFRdpnbaBmJTMBKylfruJivjaDMzh5szNBN
oOgfOLs0ggJxBp3Flc9xOcnMEmGAQJ2/PYoa22VVGshIAMGhcsgyzK+TDPGbjAv6o/76+iTIpS
VOWXSCRNryg3yneYP4QgW2UIM6idPaLPn+HlI0loyflXJqE2EH4SGoxm/oKMBgfn3zGPQyrqkV
D5po/NrDolsnGhNrhYxwYUqQrcp6Ow011cj7ptdfwkjyDFfANdd2FNUwEAAA==''':
(
'''SOLUTION:Double Bonds,Zig,60-1-20,Cycles
COMPONENT:'drag-starter-reactor',2,0,''
MEMBER:'instr-start',-90,0,128,1,6,0,0
MEMBER:'instr-start',0,0,32,0,1,0,0
MEMBER:'feature-bonder',-1,0,1,3,1,0,0
MEMBER:'feature-bonder',-1,0,1,2,1,0,0
MEMBER:'feature-bonder',-1,0,1,1,1,0,0
MEMBER:'feature-bonder',-1,0,1,9,7,0,0
MEMBER:'instr-grab',-1,1,32,1,1,0,0
MEMBER:'instr-arrow',0,0,16,1,1,0,0
MEMBER:'instr-bond',-1,0,32,2,1,0,0
MEMBER:'instr-rotate',-1,0,32,3,1,0,0
MEMBER:'instr-grab',-1,2,32,6,1,0,0
MEMBER:'instr-arrow',180,0,16,6,1,0,0
MEMBER:'instr-output',-1,0,32,4,1,0,0
MEMBER:'instr-arrow',0,0,64,1,1,0,0
MEMBER:'instr-grab',-1,1,128,1,1,0,0
MEMBER:'instr-input',-1,0,128,1,2,0,0
MEMBER:'instr-input',-1,0,128,2,1,0,0
MEMBER:'instr-input',-1,0,128,1,4,0,0
MEMBER:'instr-grab',-1,2,128,6,1,0,0
MEMBER:'instr-arrow',180,0,64,6,1,0,0
MEMBER:'instr-input',-1,0,128,4,1,0,0
MEMBER:'instr-rotate',-1,0,128,5,1,0,0
MEMBER:'instr-input',-1,0,128,1,5,0,0
MEMBER:'instr-bond',-1,0,32,5,1,0,0
MEMBER:'instr-bond',-1,0,128,1,3,0,0
MEMBER:'instr-bond',-1,0,128,3,1,0,0
PIPE:0,4,1
PIPE:1,4,2''',)
}

infinite_loops = (
('''H4sIAG36Zl4A/5WPzQqDMBCEX6XsWSF6En0Ie27pIdWNhsZEkg3Uin32JlpKf069LMzHMDszg9
Sjp/RmNDooZ2DxrCzI4wyDUdh4hVBCfZ061FV9Z3mVZUXOqjwrGIMEGuM1QZnly2lZEjCefjP/
CmIxpucuVdx2mG6BUAquHCZwNrpFmz7NbHM61M7Ylyci4R1+EjcqSfQFCRWOxr5jmsZY1KJDbp
s+NNN8WKuL3Z7rhl/CsgS4pz7+hIPsgmylEDJspCm0Wh4gLNOZWwEAAA==''',
'''SOLUTION:Of Pancakes,Zig,107-1-7
COMPONENT:'custom-research-reactor',2,0,''
MEMBER:'instr-start',0,0,128,0,1,0,0
MEMBER:'instr-start',180,0,32,0,7,0,0
MEMBER:'instr-arrow',0,0,64,2,1,0,0
MEMBER:'instr-arrow',180,0,64,6,1,0,0
MEMBER:'instr-input',-1,0,128,1,1,0,0
MEMBER:'instr-grab',-1,1,128,2,1,0,0
MEMBER:'instr-rotate',-1,0,128,4,1,0,0
PIPE:0,4,1
PIPE:1,4,2'''),
)
