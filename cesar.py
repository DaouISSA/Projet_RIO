def cesar_chiffre(x,k) :
    return (x+k)%26

def cesar_dechiffre(x,k) :
    return (x-k)%26

def cesar(mot,k):
    mot_crypte= []
    for lettre in mot: 
        nb= ord(lettre)-65
        let=cesar_chiffre(nb,k)
        c=chr(let+65)
        mot_crypte.append(c)
    mot_crypte="".join(mot_crypte)
    return mot_crypte
    
if __name__== "__main__":
    print(cesar("COUCOU",11))
    print("Issa")
        