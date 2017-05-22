//
//  RSSAppDelegateService.m
//  RSS
//
//  Created by 李占昆 on 17/5/23.
//  Copyright © 2017年 kkk. All rights reserved.
//

#import "RSSAppDelegateService.h"
#import "AppDelegate.h"

@implementation RSSAppDelegateService

+(void)load {
    [super load];
    
    __block id observer =
    [[NSNotificationCenter defaultCenter]
     addObserverForName:UIApplicationDidFinishLaunchingNotification
     object:nil
     queue:nil
     usingBlock:^(NSNotification *note) {
         NSLog(@"note == %@", note);
//         [self setup]; // Do whatever you want
         [[NSNotificationCenter defaultCenter] removeObserver:observer];
     }];
}

+ (RSSAppDelegateService*)sharedApplication
{
    static dispatch_once_t pred = 0;
    __strong static id sharedObject = nil;
    dispatch_once(&pred, ^{
        sharedObject = [[self alloc] init];
    });
    
    return sharedObject;
}

@end
