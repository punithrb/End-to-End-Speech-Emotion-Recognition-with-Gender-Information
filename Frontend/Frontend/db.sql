drop database if exists emotion_recognition;
create database emotion_recognition;
use emotion_recognition;

create table register (
    id int primary key auto_increment, 
    name varchar(225),  
    email varchar(225), 
    pwd varchar(225),
    cpwd varchar(225)
    )
