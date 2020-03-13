#!/usr/bin/env python
# -*- coding: utf-8 -*-

from timeit import timeit
import unittest

from spacechem import game, level, solution

class TestGame(unittest.TestCase):
    def test_pancakes_rotate(self):
        level_code = '''H4sIAG36Zl4A/5WPzQqDMBCEX6XsWSF6En0Ie27pIdWNhsZEkg3Uin32JlpKf069LMzHMDszg9
Sjp/RmNDooZ2DxrCzI4wyDUdh4hVBCfZ061FV9Z3mVZUXOqjwrGIMEGuM1QZnly2lZEjCefjP/
CmIxpucuVdx2mG6BUAquHCZwNrpFmz7NbHM61M7Ylyci4R1+EjcqSfQFCRWOxr5jmsZY1KJDbp
s+NNN8WKuL3Z7rhl/CsgS4pz7+hIPsgmylEDJspCm0Wh4gLNOZWwEAAA=='''
        solution_code = '''SOLUTION:Of Pancakes,Zig,107-1-7
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
PIPE:1,4,2
'''

        self.assertEqual(game.score_soln(level.ResearchLevel(level_code),
                                         solution.Solution(solution_code)),
                         (107, 7))

    def test_pancakes_cycles(self):
        level_code = '''H4sIAG36Zl4A/5WPzQqDMBCEX6XsWSF6En0Ie27pIdWNhsZEkg3Uin32JlpKf069LMzHMDszg9
 Sjp/RmNDooZ2DxrCzI4wyDUdh4hVBCfZ061FV9Z3mVZUXOqjwrGIMEGuM1QZnly2lZEjCefjP/
 CmIxpucuVdx2mG6BUAquHCZwNrpFmz7NbHM61M7Ylyci4R1+EjcqSfQFCRWOxr5jmsZY1KJDbp
 s+NNN8WKuL3Z7rhl/CsgS4pz7+hIPsgmylEDJspCm0Wh4gLNOZWwEAAA=='''
        solution_code = '''SOLUTION:Of Pancakes and Spaceships,Zig,45-1-14,Cycles
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
PIPE:1,4,2
'''

        self.assertEqual(game.score_soln(level.ResearchLevel(level_code),
                                         solution.Solution(solution_code)),
                         (45, 14))
        avg_time = timeit(lambda: game.score_soln(level.ResearchLevel(level_code),
                                                  solution.Solution(solution_code)),
                   number=100, globals=globals()) / 100
        print(f'Pancakes cycles ran in avg {avg_time} seconds')
        #self.assertLess(time, 0.003)

    def test_pancakes_infinite_loop_with_molecule(self):
        level_code = '''H4sIAG36Zl4A/5WPzQqDMBCEX6XsWSF6En0Ie27pIdWNhsZEkg3Uin32JlpKf069LMzHMDszg9
Sjp/RmNDooZ2DxrCzI4wyDUdh4hVBCfZ061FV9Z3mVZUXOqjwrGIMEGuM1QZnly2lZEjCefjP/
CmIxpucuVdx2mG6BUAquHCZwNrpFmz7NbHM61M7Ylyci4R1+EjcqSfQFCRWOxr5jmsZY1KJDbp
s+NNN8WKuL3Z7rhl/CsgS4pz7+hIPsgmylEDJspCm0Wh4gLNOZWwEAAA=='''
        solution_code = '''SOLUTION:Of Pancakes,Zig,107-1-7
COMPONENT:'custom-research-reactor',2,0,''
MEMBER:'instr-start',0,0,128,0,1,0,0
MEMBER:'instr-start',180,0,32,0,7,0,0
MEMBER:'instr-arrow',0,0,64,2,1,0,0
MEMBER:'instr-arrow',180,0,64,6,1,0,0
MEMBER:'instr-input',-1,0,128,1,1,0,0
MEMBER:'instr-grab',-1,1,128,2,1,0,0
MEMBER:'instr-rotate',-1,0,128,4,1,0,0
PIPE:0,4,1
PIPE:1,4,2
'''
        with self.assertRaises(game.InfiniteLoopException):
            game.score_soln(level.ResearchLevel(level_code), solution.Solution(solution_code))

    def test_bonding_symbols(self):
        level_code = '''H4sIAMa7aV4A/3WPQWvDMAyF/0rQaYME7FIYc07rIdDzbh07eImSGFwr2HKhC/nvs5sxtoZdBP
p47+lpBuOmyNUnOQygZhB53Fha32Y4k8U2WgQFr8Ze0NcvQy3l/kkIKKGl6BiU3C3vSwnyf29j
I3njsG6S+XnjTWaKvC2yuV48HB+LNazDVKW5dZGi3t2lipw56lBZ7Qes1nRQvbYBS/gg16Gvvs
X7VRnQBfI/moz6GPAvCZM1zHeQ0eJE/jfm65RbewyofTumZk6fMzk69tTFlg25gqk4pCrGDUmg
I4/5PpxMXjvT9yY9z1dQYvkCUjns8KkBAAA=
'''
        solution_code = '''SOLUTION:An Introduction to Bonding,Zig,154-1-10,Symbols
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
PIPE:1,4,2
'''

        self.assertEqual(game.score_soln(level.ResearchLevel(level_code),
                                         solution.Solution(solution_code)),
                         (154, 10))

        avg_time = timeit(lambda: game.score_soln(level.ResearchLevel(level_code),
                                                  solution.Solution(solution_code)),
                          number=100, globals=globals()) / 100
        print(f'Intro to Bonding symbols ran in avg {avg_time} seconds')
        #self.assertLess(time, 0.003)


    def test_bonding_cycles_with_minus(self):
        level_code = '''H4sIAMa7aV4A/3WPQWvDMAyF/0rQaYME7FIYc07rIdDzbh07eImSGFwr2HKhC/nvs5sxtoZdBP
p47+lpBuOmyNUnOQygZhB53Fha32Y4k8U2WgQFr8Ze0NcvQy3l/kkIKKGl6BiU3C3vSwnyf29j
I3njsG6S+XnjTWaKvC2yuV48HB+LNazDVKW5dZGi3t2lipw56lBZ7Qes1nRQvbYBS/gg16Gvvs
X7VRnQBfI/moz6GPAvCZM1zHeQ0eJE/jfm65RbewyofTumZk6fMzk69tTFlg25gqk4pCrGDUmg
I4/5PpxMXjvT9yY9z1dQYvkCUjns8KkBAAA=
'''
        solution_code = '''
SOLUTION:An Introduction to Bonding,Zig,74-1-40,Cycles!!!
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
PIPE:1,4,2
'''
        self.assertEqual(game.score_soln(level.ResearchLevel(level_code),
                                         solution.Solution(solution_code)),
                         (74, 40))

        avg_time = timeit(lambda: game.score_soln(level.ResearchLevel(level_code),
                                                  solution.Solution(solution_code)),
                          number=100, globals=globals()) / 100
        print(f'Intro to Bonding cycles ran in avg {avg_time} seconds')

    def test_double_bonds(self):
        level_code = '''
H4sIAHDCal4A/22PywrCMBBFf0Vm3UJaXEi6E/fuFRdpnbaBmJTMBKylfruJivjaDMzh5szNBN
oOgfOLs0ggJxBp3Flc9xOcnMEmGAQJ2/PYoa22VVGshIAMGhcsgyzK+TDPGbjAv6o/76+iTIpS
VOWXSCRNryg3yneYP4QgW2UIM6idPaLPn+HlI0loyflXJqE2EH4SGoxm/oKMBgfn3zGPQyrqkV
D5po/NrDolsnGhNrhYxwYUqQrcp6Ow011cj7ptdfwkjyDFfANdd2FNUwEAAA==
'''
        solution_code = '''
SOLUTION:Double Bonds,Zig,60-1-20,Cycles
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
PIPE:1,4,2
'''

        self.assertEqual(game.score_soln(level.ResearchLevel(level_code),
                                         solution.Solution(solution_code)),
                         (60, 20))

        avg_time = timeit(lambda: game.score_soln(level.ResearchLevel(level_code),
                                                  solution.Solution(solution_code)),
                   number=100, globals=globals()) / 100
        print(f'Double Bonds cycles ran in avg {avg_time} seconds')


if __name__ == '__main__':
    unittest.main()
