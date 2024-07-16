#Some arbitrary decisions I made:
#Computer always plays O and player always plays X
#The neural network outputs is how much computer wants to play in that position
#The 8 winning conditions are numbered in a specific way. You can see it in ./winning_condition_numbers.png


import pygame
import numpy as np
from nn import NeuralNetwork
from random import randint
import os

neuralNetwork = NeuralNetwork(9 , 6 , 9 , 0.3)

input_hidden_path = "./weights_input_hidden.csv"
hidden_output_path = "./weights_hidden_output.csv"

if os.path.exists(input_hidden_path) and os.path.exists(hidden_output_path):
    neuralNetwork.loadWeights()
else:
    neuralNetwork.saveWeights()

pygame.init()


class Button:
    def __init__(self, x, y, width, height, text, font, color, hover_color, text_color, corner_radius=0):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color
        self.hover_color = hover_color
        self.text = font.render(text, True, text_color)
        self.text_rect = self.text.get_rect(center=self.rect.center)
        self.text_color = text_color
        self.font = font
        self.corner_radius = corner_radius

    def draw(self, screen):
        current_color = self.color
        mouse_pos = pygame.mouse.get_pos()

        if self.rect.collidepoint(mouse_pos):
            current_color = self.hover_color

        self.draw_rounded_rect(screen, current_color, self.rect, self.corner_radius)
        screen.blit(self.text, self.text_rect)
        

    def draw_rounded_rect(self, surface, color, rect, corner_radius):
        if corner_radius > 0:
            pygame.draw.rect(surface, color, rect, border_radius=corner_radius)
        else:
            pygame.draw.rect(surface, color, rect)

    def is_clicked(self , event):
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.rect.collidepoint(event.pos):
                return True
        return False

# Define font
buttonsFont = pygame.font.SysFont(None , 30)
menuFont = pygame.font.SysFont(None , 40)
secondaryMenuFont = pygame.font.SysFont(None , 30)

colorPallette = {
    'red':(238, 66, 102),
    'blue':(94, 22, 117),
    'green':(51, 115, 87),
    'yellow':(255, 210, 63),
    'white':(220,220,220),
    'black':(0,0,0),
    'light_red':(240, 103, 132),
    'light_blue':(149, 55, 179),
    'light_green':(81, 176, 134) 
}




def drawCross(surface, color, coordinate, size, thickness=4):
    pygame.draw.line(surface, color, (coordinate[0] - size, coordinate[1] - size), (coordinate[0] + size, coordinate[1] + size), thickness)
    pygame.draw.line(surface, color, (coordinate[0] - size, coordinate[1] + size), (coordinate[0] + size, coordinate[1] - size), thickness)

def drawCircle(surface, color, coordinate, size, thickness=3):
    rect = pygame.Rect(coordinate[0] - size // 2, coordinate[1] - size // 2, size, size)
    pygame.draw.ellipse(surface, color, rect, thickness)

def drawBoard(surface):
    pygame.draw.line(surface , colorPallette['black'] , point1 , point7)
    pygame.draw.line(surface , colorPallette['black'] , point2 , point8)
    pygame.draw.line(surface , colorPallette['black'] , point3 , point4)
    pygame.draw.line(surface , colorPallette['black'] , point5 , point6)

def fillBoard(surface , boardState):
    for i in range(3):
        for j in range(3):
            if(boardState[i][j] == "X"):
                drawCross(surface , crossColor , cells[(i*3) + j].center , crossSize)
            elif(boardState[i][j] == "O"):
                drawCircle(surface , circleColor , cells[(i*3) + j].center , circleSize)

def drawText(screen, text, font, color, position):
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect(center=position)
    screen.blit(text_surface, text_rect)

def boardStateToNnInputs(boardState):
    inputs = []
    for i in range(3):
        for j in range(3):
            if(boardState[i][j] == " "):
                inputs.append(0.5)
            elif(boardState[i][j] == "X"):
                inputs.append(1)
            elif(boardState[i][j] == "O"):
                inputs.append(0)
    return inputs

def getBestPosition(inputs):
    nnOutputs = neuralNetwork.query(inputs)
    preferredPosition = np.argmax(nnOutputs)
    return preferredPosition

def nthBestChoice(arr, n):
    sortedArray = np.sort(arr.transpose())[0][::-1]
    position = np.where(arr.transpose()[0] == sortedArray[n-1])[0][0]
    return position

def gameSituation(boardState):
    winner = []
    for i in range(3):
        if(boardState[i][0] == "O" and boardState[i][1] == "O" and boardState[i][2] == "O"):
            winner = ["Computer" , i+4]
            return winner
            
        elif(boardState[i][0] == "X" and boardState[i][1] == "X" and boardState[i][2] == "X"):
            winner = ["Player" , i+4]
            return winner
        
        elif(boardState[0][i] == "O" and boardState[1][i] == "O" and boardState[2][i] == "O"):
            winner = ["Computer" , i+1]
            return winner
        
        elif(boardState[0][i] == "X" and boardState[1][i] == "X" and boardState[2][i] == "X"):
            winner = ["Player" , i+1]
            return winner

    if(boardState[0][0] == "O" and boardState[1][1] == "O" and boardState[2][2] == "O"):
        winner = ["Computer" , 7]
        return winner

    elif(boardState[0][0] == "X" and boardState[1][1] == "X" and boardState[2][2] == "X"):
        winner = ["Player" , 7]
        return winner
    
    elif(boardState[0][2] == "O" and boardState[1][1] == "O" and boardState[2][0] == "O"):
        winner = ["Computer" , 8]
        return winner
    
    elif(boardState[0][2] == "X" and boardState[1][1] == "X" and boardState[2][0] == "X"):
        winner = ["Player" , 8]
        return winner
    
    filled = True
    for i in range(3):
        if(boardState[i].count(" ") >= 1):
            filled = False
            break
    
    if(filled):
        return ["Tie" , None]
    else:
        return ["Playing" , None]
  

def setWinningLine(winner):
    global winningLine
    if(winner[0] == "Computer"):
        if(winner[1] == 1):
            winningLine = [circleColor , ((WIDTH/2) - (lineSeparation) , (HEIGHT/2) - (lineSeparation * 1.5) - 10) , ((WIDTH/2) - (lineSeparation) , (HEIGHT/2) + (lineSeparation * 1.5) + 10)]
        elif(winner[1] == 2):
            winningLine = [circleColor , ((WIDTH/2) , (HEIGHT/2) - (lineSeparation * 1.5) - 10) , ((WIDTH/2) , (HEIGHT/2) + (lineSeparation*1.5) + 10)]
        elif(winner[1] == 3):
            winningLine = [circleColor , ((WIDTH/2) + (lineSeparation) , (HEIGHT/2) - (lineSeparation * 1.5) - 10) , ((WIDTH/2) + (lineSeparation) , (HEIGHT/2) + (lineSeparation*1.5) + 10)]
        elif(winner[1] == 4):
            winningLine = [circleColor , ((WIDTH/2) - (lineSeparation * 1.5) - 10 , (HEIGHT/2) - lineSeparation) , ((WIDTH/2) + (lineSeparation * 1.5) + 10 , (HEIGHT/2) - lineSeparation)]
        elif(winner[1] == 5):
            winningLine = [circleColor , ((WIDTH/2) - (lineSeparation * 1.5) - 10 , (HEIGHT/2)) , ((WIDTH/2) + (lineSeparation * 1.5) + 10 , (HEIGHT/2))]
        elif(winner[1] == 6):
            winningLine = [circleColor , ((WIDTH/2) - (lineSeparation * 1.5) - 10 , (HEIGHT/2) + lineSeparation) , ((WIDTH/2) + (lineSeparation * 1.5) + 10 , (HEIGHT/2) + lineSeparation)]
        elif(winner[1] == 7):
            winningLine = [circleColor , ((WIDTH/2) - (lineSeparation * 1.5) - 10 , (HEIGHT/2) - (lineSeparation * 1.5) - 10) , ((WIDTH/2) + (lineSeparation * 1.5) + 10 , (HEIGHT/2) + (lineSeparation * 1.5) + 10)]
        elif(winner[1] == 8):
            winningLine = [circleColor , ((WIDTH/2) + (lineSeparation * 1.5) + 10 , (HEIGHT/2) - (lineSeparation * 1.5) - 10) , ((WIDTH/2) - (lineSeparation * 1.5) - 10 , (HEIGHT/2) + (lineSeparation * 1.5) + 10)]
        
    
    elif(winner[0] == "Player"):
        if(winner[1] == 1):
            winningLine = [crossColor , ((WIDTH/2) - (lineSeparation) , (HEIGHT/2) - (lineSeparation * 1.5) - 10) , ((WIDTH/2) - (lineSeparation) , (HEIGHT/2) + (lineSeparation * 1.5) + 10)]
        elif(winner[1] == 2):
            winningLine = [crossColor , ((WIDTH/2) , (HEIGHT/2) - (lineSeparation * 1.5) - 10) , ((WIDTH/2) , (HEIGHT/2) + (lineSeparation*1.5) + 10)]
        elif(winner[1] == 3):
            winningLine = [crossColor , ((WIDTH/2) + (lineSeparation) , (HEIGHT/2) - (lineSeparation * 1.5) - 10) , ((WIDTH/2) + (lineSeparation) , (HEIGHT/2) + (lineSeparation*1.5) + 10)]
        elif(winner[1] == 4):
            winningLine = [crossColor , ((WIDTH/2) - (lineSeparation * 1.5) - 10 , (HEIGHT/2) - lineSeparation) , ((WIDTH/2) + (lineSeparation * 1.5) + 10 , (HEIGHT/2) - lineSeparation)]
        elif(winner[1] == 5):
            winningLine = [crossColor , ((WIDTH/2) - (lineSeparation * 1.5) - 10 , (HEIGHT/2)) , ((WIDTH/2) + (lineSeparation * 1.5) + 10 , (HEIGHT/2))]
        elif(winner[1] == 6):
            winningLine = [crossColor , ((WIDTH/2) - (lineSeparation * 1.5) - 10 , (HEIGHT/2) + lineSeparation) , ((WIDTH/2) + (lineSeparation * 1.5) + 10 , (HEIGHT/2) + lineSeparation)]
        elif(winner[1] == 7):
            winningLine = [crossColor , ((WIDTH/2) - (lineSeparation * 1.5) - 10 , (HEIGHT/2) - (lineSeparation * 1.5) - 10) , ((WIDTH/2) + (lineSeparation * 1.5) + 10 , (HEIGHT/2) + (lineSeparation * 1.5) + 10)]
        elif(winner[1] == 8):
            winningLine = [crossColor , ((WIDTH/2) + (lineSeparation * 1.5) + 10 , (HEIGHT/2) - (lineSeparation * 1.5) - 10) , ((WIDTH/2) - (lineSeparation * 1.5) - 10 , (HEIGHT/2) + (lineSeparation * 1.5) + 10)]
        

def trainSequence():
    global boardState

    trainingTurn = "Computer" if randint(0,1) == 0 else "Player"
    noOfTurnsToTrain = randint(0 , 8)
    # noOfTurnsToTrain = 1


    for i in range(noOfTurnsToTrain):
        positionToPlay = randint(0 , 8)
        while (boardState[int(positionToPlay/3)][positionToPlay%3] != " "):
            positionToPlay = randint(0 , 8)
        
        if(trainingTurn == "Computer"):
            boardState[int(positionToPlay/3)][positionToPlay%3] = "O"
            trainingTurn = "Player"
        else:
            boardState[int(positionToPlay/3)][positionToPlay%3] = "X"
            trainingTurn = "Computer"
        
    if(gameSituation(boardState)[0] != "Playing" or trainingTurn != "Player"):
        boardState = [
                    [' ' , ' ' , ' '],
                    [' ' , ' ' , ' '],
                    [' ' , ' ' , ' ']
                ]
        trainSequence()





WIDTH = 800
HEIGHT = 600


screen = pygame.display.set_mode((WIDTH , HEIGHT))
pygame.display.set_caption("TIC TAC TOE")
clock = pygame.time.Clock()
frameRate = 50



mainButtonsWidth = 100
mainButtonsHeight = 40
secondaryButtonsWidth = 100
secondaryButtonsHeight = 40
mainButtonsGap = 50

backButtonWidth = 80
backButtonHeight = 30



crossSize = 30
crossColor = (3,48,31)

circleSize = 70
circleColor = (189,1,1)



lineSeparation = 100 #of the board

'''
       p1    p2
        |    |
     C1 | C2 | C3
 p3 ----+----+---- p4
        |    |
     C4 | C5 | C6
 p5 ----+----+---- p6
     C7 | C8 | C9
        |    |
       p7    p8
'''

point1 = ( (WIDTH/2) - (lineSeparation/2) , (HEIGHT/2) - (lineSeparation*1.5) )
point2 = ( (WIDTH/2) + (lineSeparation/2) , (HEIGHT/2) - (lineSeparation*1.5) )
point3 = ( (WIDTH/2) - (lineSeparation*1.5) , (HEIGHT/2) - (lineSeparation/2) )
point4 = ( (WIDTH/2) + (lineSeparation*1.5) , (HEIGHT/2) - (lineSeparation/2) )
point5 = ( (WIDTH/2) - (lineSeparation*1.5) , (HEIGHT/2) + (lineSeparation/2) )
point6 = ( (WIDTH/2) + (lineSeparation*1.5) , (HEIGHT/2) + (lineSeparation/2) )
point7 = ( (WIDTH/2) - (lineSeparation/2) , (HEIGHT/2) + (lineSeparation*1.5) )
point8 = ( (WIDTH/2) + (lineSeparation/2) , (HEIGHT/2) + (lineSeparation*1.5) )

cell1 = pygame.Rect( (WIDTH/2) - (lineSeparation*1) - (lineSeparation/2) , (HEIGHT/2) - (lineSeparation*1) - (lineSeparation/2) , lineSeparation , lineSeparation )
cell2 = pygame.Rect( (WIDTH/2) - (lineSeparation*0) - (lineSeparation/2) , (HEIGHT/2) - (lineSeparation*1) - (lineSeparation/2) , lineSeparation , lineSeparation )
cell3 = pygame.Rect( (WIDTH/2) + (lineSeparation*1) - (lineSeparation/2) , (HEIGHT/2) - (lineSeparation*1) - (lineSeparation/2) , lineSeparation , lineSeparation )
cell4 = pygame.Rect( (WIDTH/2) - (lineSeparation*1) - (lineSeparation/2) , (HEIGHT/2) - (lineSeparation*0) - (lineSeparation/2) , lineSeparation , lineSeparation )
cell5 = pygame.Rect( (WIDTH/2) - (lineSeparation*0) - (lineSeparation/2) , (HEIGHT/2) - (lineSeparation*0) - (lineSeparation/2) , lineSeparation , lineSeparation )
cell6 = pygame.Rect( (WIDTH/2) + (lineSeparation*1) - (lineSeparation/2) , (HEIGHT/2) - (lineSeparation*0) - (lineSeparation/2) , lineSeparation , lineSeparation )
cell7 = pygame.Rect( (WIDTH/2) - (lineSeparation*1) - (lineSeparation/2) , (HEIGHT/2) + (lineSeparation*1) - (lineSeparation/2) , lineSeparation , lineSeparation )
cell8 = pygame.Rect( (WIDTH/2) - (lineSeparation*0) - (lineSeparation/2) , (HEIGHT/2) + (lineSeparation*1) - (lineSeparation/2) , lineSeparation , lineSeparation )
cell9 = pygame.Rect( (WIDTH/2) + (lineSeparation*1) - (lineSeparation/2) , (HEIGHT/2) + (lineSeparation*1) - (lineSeparation/2) , lineSeparation , lineSeparation )

cells = [cell1 , cell2 , cell3 , cell4 , cell5 , cell6 , cell7 , cell8 , cell9]


boardState = [
                [' ' , ' ' , ' '],
                [' ' , ' ' , ' '],
                [' ' , ' ' , ' ']          
            ]





menu = "Main"

trainButton = Button( (WIDTH/2) - (mainButtonsGap/2) - (mainButtonsWidth) , (HEIGHT/2) - (mainButtonsHeight/2) - (mainButtonsGap/2) , mainButtonsWidth , mainButtonsHeight , "Train" , buttonsFont , colorPallette['green'] , colorPallette['light_green'] , colorPallette['black'] , 3)
playButton = Button( (WIDTH/2) + (mainButtonsGap/2) , (HEIGHT/2) - (mainButtonsHeight/2) - (mainButtonsGap/2) , mainButtonsWidth , mainButtonsHeight , "Play" , buttonsFont , colorPallette['green'] , colorPallette['light_green'] , colorPallette['black'] , 3)
quitButton = Button( (WIDTH/2) - (secondaryButtonsWidth/2) , (HEIGHT/2) + (mainButtonsGap/2) , secondaryButtonsWidth , secondaryButtonsHeight , "Quit" , buttonsFont , colorPallette['red'] , colorPallette['light_red'] , colorPallette['black'] , 3)

backButton = Button(0 , 0 , backButtonWidth , backButtonHeight , "Back" , buttonsFont , colorPallette['red'] , colorPallette['light_red'] , colorPallette['black'])

winningLine = []

turn = "Computer" if randint(0,1) == 0 else "Player"


playing = True
while playing:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            playing = False
        
        if(menu == "Main"):
            if(trainButton.is_clicked(event)):
                menu = "Train"
                boardState = [
                    [' ' , ' ' , ' '],
                    [' ' , ' ' , ' '],
                    [' ' , ' ' , ' ']
                ]
                trainSequence()


            
            if(playButton.is_clicked(event)):
                menu = "Play"
            
            if(quitButton.is_clicked(event)):
                playing = False
        

        if(menu != "Main"):
            if(backButton.is_clicked(event)):
                menu = "Main"
                boardState = [
                    [' ' , ' ' , ' '],
                    [' ' , ' ' , ' '],
                    [' ' , ' ' , ' ']
                ]
                turn = "Computer" if randint(0,1) == 0 else "Player"
        

        if(menu == "Play" and turn == "Player" and event.type == pygame.MOUSEBUTTONDOWN):
            for index, cell in enumerate(cells):
                if cell.collidepoint(event.pos) :
                    if(boardState[int(index/3)][index%3] == " " and gameSituation(boardState)[0] == "Playing"): #if the cell is empty AND the game is "playing"
                        boardState[int(index/3)][index%3] = "X"
                        turn = "Computer"

                        if(gameSituation(boardState)[0] == "Player" or gameSituation(boardState)[0] == "Computer"):
                            setWinningLine(gameSituation(boardState))

        elif(menu == "Train" and event.type == pygame.MOUSEBUTTONDOWN):
            for index, cell in enumerate(cells):
                if cell.collidepoint(event.pos):
                    if(boardState[int(index/3)][index%3] == " " and gameSituation(boardState)[0] == "Playing"):


                        inputs = boardStateToNnInputs(boardState)
                        targets = [0,0,0,0,0,0,0,0,0]
                        targets[index] = 1
                        
                        neuralNetwork.train(inputs , targets)

                        neuralNetwork.saveWeights()
                        # print(inputs)
                        # print(targets)

                        boardState = [
                            [' ' , ' ' , ' '],
                            [' ' , ' ' , ' '],
                            [' ' , ' ' , ' ']
                        ]
                        trainSequence()

                

    

    screen.fill(colorPallette['white'])
    if(menu == "Main"):
        trainButton.draw(screen)
        playButton.draw(screen)
        quitButton.draw(screen)

        drawText(screen , "Main Menu" , menuFont , colorPallette['black'] , (WIDTH/2 , 30))

        


    elif(menu == "Train"):
        backButton.draw(screen)
        drawText(screen , "Train" , menuFont , colorPallette['black'] , (WIDTH/2 , 30))
        drawText(screen , "What would you do in this position as 'X'?" , secondaryMenuFont , colorPallette['black'] , (WIDTH/2 , 60))
        drawBoard(screen)
        fillBoard(screen , boardState)




    elif(menu == "Play"):
        backButton.draw(screen)
        drawText(screen , "Play" , menuFont , colorPallette['black'] , (WIDTH/2 , 30))
        drawBoard(screen)
        fillBoard(screen , boardState)
        if((gameSituation(boardState)[0] == "Player" or gameSituation(boardState)[0] == "Computer") and len(winningLine) != 0):
            pygame.draw.line(screen , winningLine[0] , winningLine[1] , winningLine[2])

        if(turn == "Computer"):
            inputs = boardStateToNnInputs(boardState)
            outputs = neuralNetwork.query(inputs)
            for i in range(9):
                positionToPlay = nthBestChoice(outputs , i+1)
                if(boardState[int(positionToPlay/3)][positionToPlay%3] == " "):
                    break

            if(gameSituation(boardState)[0] == "Playing"):
                boardState[int(positionToPlay/3)][positionToPlay%3] = "O"
                turn = "Player"

                if(gameSituation(boardState)[0] == "Player" or gameSituation(boardState)[0] == "Computer"):
                    setWinningLine(gameSituation(boardState))
            
            
        



    clock.tick(frameRate)
    pygame.display.flip()



pygame.quit()



